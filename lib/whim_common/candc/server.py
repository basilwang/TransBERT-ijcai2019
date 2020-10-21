"""
Tool for running lots of C&C parsers at once. Sets up a load of Soap servers and fires off parse jobs to
them as necessary.

"""
__author__ = 'Mark Granroth-Wilding'

from threading import Thread, Event
from whim_common.candc.monitor import PoolMonitorThread
from Queue import Queue, Empty
import subprocess
import os
import time
import sys
import tempfile
import select
import logging
import math
import itertools


CANDC_BIN_DIR = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 "..", "..", "..", "..", "lib", "candc-1.00", "bin"))
CANDC_SERVER_COMMAND = os.path.join(CANDC_BIN_DIR, "soap_server")
CANDC_CLIENT_COMMAND = os.path.join(CANDC_BIN_DIR, "soap_client")
CANDC_MODELS_DIR = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 "..", "..","..","..", "lib","candc-1.00","models"))


class CandCPool(object):
    def __init__(self, host="127.0.0.1", ports=[9000], name="candc-pool", callback=lambda filename, output: None, extra_args=[]):
        self.logger = logging.getLogger("candc")
        self.host = host
        self.ports = ports
        self._extra_args = extra_args
        self._callback = callback
        
        self.name = name
        # Create a queue through which we'll feed jobs to the workers
        self._queue = Queue()

        # Start up a load of Soap servers
        self.workers = []
        for port in ports:
            # Start a server
            self.start_worker(port)

        # Start a monitor thread so we can check up on things
        self.monitor_thread = PoolMonitorThread(name, self)
        self.monitor_thread.start()
        
        # Keep track of how many jobs have been put in the queue
        self.jobs_added = 0

    def parse_file(self, filename):
        """
        Add a file to the queue for parsing. Once the parse job is complete, the pool's callback will be
        called on the result.

        """
        self._queue.put(filename)
        self.jobs_added += 1

    def join(self):
        """
        Called when all items have been put in the queue to block until they've all been processed.

        """
        # Wait until all the jobs are done
        # Drop out if all the workers die
        while self._queue.unfinished_tasks and any(worker.process.alive() for worker in self.workers):
            time.sleep(1)
        if self._queue.unfinished_tasks:
            raise CandCServerError("Worker threads all died before processing was complete")

    def close(self):
        """
        Shut down the servers and the worker threads. Typically called from join_and_close() when everything's
        done, but might be called before things are finished.

        """
        # Shut down the C&C servers
        self.stop_servers()
        # Shut down the worker threads
        self.stop_workers()

    def join_and_close(self):
        """
        Do a join() and then close down all the worker threads and Soap servers.

        """
        # Wait until all the jobs in the queue are complete
        self.join()
        # Shut everything down
        self.close()
    
    def start_worker(self, port):
        # Start a server
        log(self.logger, "Starting soap server on %s:%s" % (self.host, port))
        process = CandCProcess(self.host, port, extra_args=self._extra_args)
        # Set up a worker thread to send requests to this server
        worker = CandCWorkerThread(process, self._queue, callback=self._callback)
        worker.start()
        self.workers.append(worker)

    def stop_workers(self):
        """
        Stop all running workers to shut down the pool.
        
        """
        # Tell the threads issuing the client calls to stop
        for worker in self.workers:
            worker.stop()
        # Wait until the workers stop before returning (they can take up to 0.5s to stop)
        log(self.logger, "Waiting for worker threads to stop")
        for worker in self.workers:
            worker.join()
        # Stop the monitor thread too
        log(self.logger, "Stopping monitor thread")
        self.monitor_thread.stop()
        self.monitor_thread.join()
        log(self.logger, "All workers stopped")
    
    def stop_worker(self):
        """
        Choose a worker from the list and ask it to stop. It will 
        finish its current job, then become inactive.
        Returns True if a thread was stopped, False if there were none 
        available to stop.
        
        """
        for worker in reverse(self.workers):
            # If this worker's already stopping or stopped, stop another one
            if worker.isAlive() and not worker.stop_pending():
                worker.stop()
                self.workers.remove(worker)
                return True
        return False

    @property
    def threads(self):
        return len(self.ports)
    
    @property
    def completed_jobs(self):
        return self.jobs_added - self._queue.qsize() - self._queue.unfinished_tasks


class CandCWorkerThread(Thread):
    """
    Thread for sending jobs to a C&C soap server. Filenames are fed to the thread through a queue. It parses
    the files and get the next one from the queue indefinitely. The thread blocks if the queue is empty until
    something appears on it.

    An optional callback processes the result of each parse job. It takes two arguments: the input filename
    and the parser's output (a CandCOutput object).

    """
    STATE_WAITING = "waiting"
    STATE_PARSING = "parsing"
    STATE_INIT = "init"
    STATE_STOPPED = "stopped"

    def __init__(self, process, queue, callback=lambda filename,output: None, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        # Report state as initializing
        self.state = CandCWorkerThread.STATE_INIT

        # Program will exit if only worker threads are running
        self.daemon = True

        self.process = process
        self.queue = queue
        self._callback = callback
        self.logger = logging.getLogger("candc")

        self._stop = Event()
        self._stopped = False

    def run(self):
        # Keep track of how many times we've gone round waiting for something in the queue
        waiting_loops = 0
        # Report state as waiting until we get a job to run
        self.state = CandCWorkerThread.STATE_WAITING

        while not self._stop.is_set():
            if not self.process.alive():
                # Oh no! The server's fallen over
                log(self.logger, "C&C soap server has stopped running: exiting worker thread", error=True)
                break
            try:
                # Wait until something appears in the queue for us to do
                # Time out every now and again and just go round the loop to check we're not supposed to be stopping
                filename = self.queue.get(block=True, timeout=0.5)
                # Now we've got a job, report state as parsing until it's done
                self.state = CandCWorkerThread.STATE_PARSING

                try:
                    waiting_loops = 0
                    # We've got a job: call the parser on this file
                    output = self.process.parse_file(filename)
                    # Call the callback when we've got the result
                    self._callback(filename, output)
                except subprocess.CalledProcessError, e:
                    log(self.logger, "C&C client returned with an error: %s" % e, error=True)
                    log(self.logger, self.process.get_stderr(), error=True)
                except Exception, e:
                    # Catch any exceptions so that the worker can continue working on other jobs afterwards
                    self.logger.exception("Error in C&C worker thread")
                finally:
                    # Read off any stderr/stdout output to flush the pipes
                    self.process.clear_pipes()
                    self.queue.task_done()
                    # No longer parsing: report states as waiting
                    self.state = CandCWorkerThread.STATE_WAITING
            except Empty:
                waiting_loops += 1

                # Log something if we've been waiting (more than) 2 mins without getting a job
                if waiting_loops % 120 == 0:
                    log(self.logger, "Server %s still waiting for a job" % self.process.name)
        self._stopped = True
        self.state = CandCWorkerThread.STATE_STOPPED
        log(self.logger, "Worker thread %s terminated, stopping C&C server" % self.process.name)
        self.process.close()

    def stop(self):
        """
        Gracefully stop the thread from running once it's finished the task it's currently working on (if any).

        """
        self._stop.set()
    
    def stop_pending(self):
        """
        True if the thread has been asked to stop, but is still waiting 
        on the current job to finish.
        
        """
        return self._stop.isSet() and not self._stopped


class CandCProcess(object):
    """
    A C&C Soap server, running in the background, and a mechanism to issue parse requests via the Soap client
    to that server.

    """
    def __init__(self, host="127.0.0.1", port=9000, extra_args=[]):
        if extra_args is None:
            extra_args = []
        self._host = host
        self._port = port
        # Start up a Soap server in the background
        server_args = [CANDC_SERVER_COMMAND,
                       "--models", CANDC_MODELS_DIR,
                       "--server", "%s:%d" % (host, port)] + extra_args
        self._server_process = subprocess.Popen(server_args,
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
        # The server outputs a line to stderr when it's ready, so we should wait for this before continuing
        err = read_if_available(self._server_process.stderr, timeout=10.0)
        # Check that the server's running nicely
        if err is None or not err.startswith("waiting for connections"):
            print >>sys.stderr, "Error starting Soap server: (%s)" % " ".join(server_args)
            if err is None:
                print >>sys.stderr, "No response received within 10 seconds"
            else:
                print >>sys.stderr, "Error: %s" % err
            raise CandCServerError("server startup failed")
    
    @property
    def name(self):
        return "c&c/%s:%s" % (self._host, self._port)

    def parse_lines(self, lines):
        """
        Run the Soap client to parse a list of sentences.

        """
        # Put the lines in a temporary file so we can pipe them to the client
        with tempfile.NamedTemporaryFile(delete=True) as input_file:
            # Write the data to the temporary file
            input_file.write("".join("%s\n" % line for line in lines))
            input_file.flush()
            input_file.seek(0)
            # Run the soap client to parse the lines
            raw_output = self._run_client(self._host, self._port, input_file)
        # Parse the output
        return CandCOutput.read_output(raw_output)

    def parse_file(self, filename, split=400):
        """
        Run the Soap client to parse all the sentences in a file.
        
        If the file is more than C{split} lines long, breaks it into 
        multiple files and parses them separately (default 400). This 
        helps to avoid buffering problems, so is advisable. Set to 0 
        to do no splitting.

        """
        if split:
            # Check file length
            with open(filename, 'r') as input_file:
                lines = len(list(input_file))
                
            if lines > split:
                raw_outputs = []
                with open(filename, 'r') as input_file:
                    # This file is too long: split it into multiple
                    for part in range(int(math.ceil(float(lines) / split))):
                        with tempfile.NamedTemporaryFile('w') as part_file:
                            # Write this partition of the input into a temp file
                            part_file.writelines(itertools.islice(input_file, split))
                            part_file.seek(0)
                            # Run the parser on this partial file
                            raw_outputs.append(self._run_client(self._host, self._port, part_file))
                            # This is critical! Otherwise the output pipes 
                            # get full while parsing long files
                            self.clear_pipes()
                # Combine the output from these parses
                return CandCOutput.read_output("\n\n".join(raw_outputs))
        
        with open(filename, 'r') as input_file:
            # Run the soap client to parse the lines
            raw_output = self._run_client(self._host, self._port, input_file)
        # Parse the output
        return CandCOutput.read_output(raw_output)

    def _run_client(self, host, port, input_file):
        # Run the soap client
        return subprocess.check_output([CANDC_CLIENT_COMMAND, "--url", "http://%s:%d" % (host, port)],
                                             stdin=input_file)

    def get_stderr(self):
        return non_block_read(self._server_process.stderr)

    def clear_pipes(self):
        """
        Flush any process output out of the stderr and stdout pipes. This should be done when you know
        you won't want the output again, e.g. once a parse job has completed successfully.

        """
        pipes = [self._server_process.stderr, self._server_process.stdout]
        # The pipes might be None if the server's shut down
        pipes = [p for p in pipes if p is not None]
        # Read anything that's there to remove it
        for pipe in pipes:
            non_block_read(pipe)

    def alive(self):
        return self._server_process.poll() is None

    def close(self):
        self._server_process.terminate()


class CandCServerError(Exception):
    pass


class CandCOutput(object):
    """
    Simple wrapper around C&C's output string to pull out the different bits.

    """
    def __init__(self, lines):
        self.data = "\n".join(lines)

        ### Parse the parser output
        # Pull out max 1 pos line -- marked by <c>
        pos_lines = [line.partition("<c> ")[2] for line in lines if line.startswith("<c> ")]
        self.pos = pos_lines[0] if len(pos_lines) else None
        # Pull out all dependency lines -- those contained in ()s
        self.dependencies = [line for line in lines if line.startswith("(") and line.endswith(")")]

    def __str__(self):
        dep_lines = "\n".join(self.dependencies)
        pos_line = "\n<c> %s" % self.pos if self.pos else ""
        return "%s%s" % (dep_lines, pos_line)

    @staticmethod
    def read_output(data):
        # Split up the data on blank lines
        outputs_lines = [o.split("\n") for o in data.split("\n\n")]
        # Ignore blank lines and comment lines
        outputs_lines = [[line for line in lines if line and not line.startswith("#")]
                            for lines in outputs_lines]
        outputs_lines = [ls for ls in outputs_lines if len(ls)]
        # Parse each as an individual parser output
        return [CandCOutput(output) for output in outputs_lines]


def read_if_available(pipe, timeout=0.2):
    readable = select.select([pipe], [], [], timeout)[0]
    if len(readable):
        return pipe.readline()
    else:
        return None


def non_block_read(output):
    """
    Read everything from the pipe until there's nothing left.

    """
    while read_if_available(output, timeout=0) is not None:
        pass


def log(logger, message, error=False):
    if logger is not None:
        if error:
            logger.error(message)
        else:
            logger.info(message)
