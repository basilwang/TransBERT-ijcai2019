"""
Monitor a C&C parser pool. A monitor thread is set going as part of the 
server's thread pool and sits in the background waiting to return 
information about the running jobs. A monitor client can be used to 
query the pool and find out information about what's currently going on.

"""
import argparse
import os
import sys
from threading import Thread, Event
import time


class PoolMonitorClient(object):
    """
    Client to poke into a running C&C pool and see what its doing. 
    Also allows you to change the number of C&C servers being used.
    
    """
    def __init__(self, name):
        self.name = name

    def _send_command(self, *args):
        ### Send the command
        with open("%s-monitor-input" % self.name, 'w') as output_pipe:
            output_pipe.write("%s\n" % " ".join(args))
        
        ### Get a response
        # Open pipe to read from the monitor thread (running in another process)
        # This will block until something is written
        input_pipe = open("%s-monitor-output" % self.name, 'r')
        try:
            # Make sure we've actually got a response before returning it
            response = ""
            while not response:
                time.sleep(0.1)
                response = input_pipe.readline().strip("\n")
        finally:
            input_pipe.close()
        
        # Split up lines within the response
        response = response.replace("//", "\n")
        return response

    def ping(self):
        print "Ping..."
        r = self._send_command("PING")
        print "Response: %s" % r
    
    def workers(self):
        print "Querying worker threads"
        print self._send_command("WORKERS")
    
    def queue(self):
        print "Job queue status:"
        print self._send_command("QUEUE")
    
    def stop_workers(self, num=1):
        print "Stopping %d worker thread(s)" % num
        stopped = self._send_command("STOP %d" % num)
        print "Stopped: %s" % stopped
        
    def _quit(self):
        """
        Causes the monitor thread to terminate. This should only be 
        called by the monitor thread object itself (from the main 
        thread). Don't call it from the client program!
        
        """
        ### Send the QUIT command and don't get a response
        with open("%s-monitor-input" % self.name, 'w') as output_pipe:
            output_pipe.write("QUIT\n")


class PoolMonitorThread(Thread):
    """
    Thread that runs in the background of a C&C pool to allow the 
    monitor client to get into it.
    
    """
    def __init__(self, name, pool, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        self.name = name
        self.workers = pool.workers
        self.queue = pool._queue
        self.pool = pool

        self._stop = Event()

    def run(self):
        input_pipe_file = "%s-monitor-input" % self.name
        output_pipe_file = "%s-monitor-output" % self.name
        # If these pipes already exist, it's probably from a previous run that didn't clean up after itself
        if os.path.exists(input_pipe_file):
            os.unlink(input_pipe_file)
        # Open a pipe so that another process can send signals to us
        os.mkfifo(input_pipe_file)

        if os.path.exists(output_pipe_file):
            os.unlink(output_pipe_file)
        # Open another pipe to send the results back
        os.mkfifo(output_pipe_file)
        
        try:
            # This will now block until something opens the pipe for writing
            with open(input_pipe_file, 'r') as input_pipe:
                # Keep reading from the pipe whenever a command is sent
                while not self._stop.isSet():
                    time.sleep(0.2)
                    command = input_pipe.readline().strip("\n")
                    command, __, args = command.partition(" ")
                    args = args.split()

                    if command:
                        # Repond to the command in some way
                        response = "unknown command"

                        rlines = []
                        #### Process commands
                        if command == "PING":
                            rlines.append("PONG")
                        elif command == "WORKERS":
                            # Send some info about each worker thread
                            for worker in self.workers:
                                rlines.append("%s: %s (%s)" % \
                                        (worker.name, 
                                         "active" if worker.isAlive() else "dead",
                                         worker.state))
                        elif command == "QUEUE":
                            if self.queue.empty():
                                rlines.append("Queue empty")
                            else:
                                rlines.append("Jobs in queue: %d" % self.queue.qsize())
                                if self.queue.full():
                                    rline.append("Queue full")
                            rlines.append("Jobs taken but unfinished: %d" % (self.queue.unfinished_tasks - self.queue.qsize()))
                            rlines.append("Completed: %d" % self.pool.completed_jobs)
                        elif command == "STOP":
                            # Stop a number of worker threads
                            num = int(args[0])
                            stopped = 0
                            for i in range(num):
                                stopped += 1 if self.pool.stop_worker() else 0
                            rlines.append("%d" % stopped)
                        elif command == "START":
                            # Start a worker thread
                            port = int(args[0])
                            self.pool.start_worker(port)
                            rlines.append("Started new C&C worker on port %d" % port)
                        elif command == "QUIT":
                            # Quit the monitor thread
                            break
                        
                        # Send a response
                        # Put it all on one line to make communication easier
                        response = "//".join(rlines)
                        with open(output_pipe_file, 'w') as output_pipe:
                            output_pipe.write("%s\n" % response)
        finally:
            # Get rid of the FIFOs
            os.unlink("%s-monitor-input" % self.name)
            os.unlink("%s-monitor-output" % self.name)

    def stop(self):
        """ Gracefully stop the thread from running immediately. """
        client = PoolMonitorClient(self.name)
        client._quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a lot of files in a directory's subtree using C&C")
    parser.add_argument("--name", help="Name of the parser pool", default="candc-pool")
    parser.add_argument("command", help="Command to issue to the monitor (use 'help' to see a list)", nargs="+")
    opts = parser.parse_args()

    client = PoolMonitorClient(opts.name)
    command = opts.command

    if command[0].upper() == "HELP":
        print "Commands:"
        print "  HELP      show this list"
        print "  PING      just ping the monitor thread of the running C&C pool to check it's responding"
        print "  WORKERS   show information about each worker thread"
        print "  QUEUE     status of job queue"
        print "  -         stop a worker thread (multiple -s will stop the corresponding number of threads)"
        print "  +<port>   start a new worker, using port <port>"
    elif command[0].upper() == "PING":
        client.ping()
    elif command[0].upper() == "WORKERS":
        client.workers()
    elif command[0].upper() == "QUEUE":
        client.queue()
    elif all(l == "-" for l in command[0]):
        client.stop_workers(len(command[0]))
    elif command[0].startswith("+"):
        client.start_worker(int(command[0][1:]))
    else:
        print >>sys.stderr, "Unknown command: %s" % command[0]
