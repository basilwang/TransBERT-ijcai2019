"""
Parse a large number of files in a directory and write the parser output to another directory.
Uses the C&C soap server to avoid having to reload the models for every parse job.

"""
from whim_common.candc.server import CandCPool
from progressbar import ProgressBar, Percentage, Bar, RotatingMarker, ETA
import os
import argparse
import sys
import logging


def parse_directory(input_dir, dep_output_dir, pos_output_dir=None, host=None, ports=None, include=None,
                    extra_args=[], show_progress=False, pool_name=None):
    pbar = ProgressBar(widgets=[Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()])

    # Prepare a callback (in a closure) to handle the parser's output
    def __get_callback(_in_dir, _dep_dir, _pos_dir, progress):
        def _callback(input_filename, output):
            """ Callback to process the parser's output. """
            # Get the filename relative to the input dir
            rel_filename = os.path.relpath(input_filename, _in_dir)
            
            # Output dependencies to a file
            dep_filename = os.path.join(_dep_dir, rel_filename)
            if not os.path.exists(os.path.dirname(dep_filename)):
                os.makedirs(os.path.dirname(dep_filename))
            with open(dep_filename, 'w') as dep_file:
                dep_file.write("\n\n".join(
                    "\n".join(sent_out.dependencies)
                        for sent_out in output
                ))
                
            # Output the tags to another file
            tag_filename = os.path.join(_pos_dir, rel_filename)
            if not os.path.exists(os.path.dirname(tag_filename)):
                os.makedirs(os.path.dirname(tag_filename))
            with open(tag_filename, 'w') as pos_file:
                pos_file.write("\n".join([sent_out.pos for sent_out in output]))
            
            if show_progress:
                # Update the progress bar
                if progress.currval < progress.maxval:
                    # Make sure we don't overshoot...just in case!
                    progress.update(progress.currval+1)

        return _callback

    # Create output directories if they don't exist
    if not os.path.exists(dep_output_dir):
        os.makedirs(dep_output_dir)
    if pos_output_dir is not None and not os.path.exists(pos_output_dir):
        os.makedirs(pos_output_dir)
        
    pool_kwargs = {}
    if pool_name is not None:
        pool_kwargs["name"] = pool_name
    if ports is not None:
        pool_kwargs["ports"] = ports
    if host is not None:
        pool_kwargs["host"] = host

    pool = CandCPool(callback=__get_callback(input_dir, dep_output_dir, pos_output_dir, pbar),
                     extra_args=extra_args, **pool_kwargs)

    try:
        # Prepare a list of files
        files = sum([[(directory,filename) for filename in filenames]
                     for directory, directories, filenames in os.walk(input_dir)], [])

        # Add all the files in the input directory to the queue
        files_to_parse = []
        for directory, filename in files:
            # Allow the include list to be given as paths (relative to input dir) or basenames
            file_path = os.path.relpath(os.path.join(directory, filename), input_dir)
            if include is not None and filename not in include and file_path not in include:
                # If a list of files was given, skip any not in the list
                continue
            files_to_parse.append(os.path.join(directory, filename))
        print >>sys.stderr, "Parsing all files in %s (%d files)" % (input_dir, len(files_to_parse))
            
        if show_progress:
            pbar.maxval = len(files_to_parse)
            pbar.start()
        
        # Put the files into the pool's queue
        for f in files_to_parse:
            pool.parse_file(f)
        # Wait till they're all done
        pool.join()

        if show_progress:
            pbar.finish()
        print >>sys.stderr, "Parsing completed, shutting down pool"
    finally:
        pool.stop_workers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a lot of files in a directory's subtree using C&C")
    parser.add_argument("input_dir", help="Root directory to read files from")
    parser.add_argument("--dep-dir", help="Dir to output dependency parses to (default: deps)", default="deps")
    parser.add_argument("--pos-dir", help="Dir to output parts of speech to (default: pos)", default="pos")
    parser.add_argument("--pool-name", help="Name to use to identify the pool. This affects the " \
                        "naming of pipes for communication with the pool. Default: 'candc-pool'")
    parser.add_argument("--host", help="Hostname to use to run C&C's Soap server (default: 127.0.0.1)",
                        default="127.0.0.1")
    parser.add_argument("--port", help="First port number for Soap servers (default: 9000)", default=9000, type=int)
    parser.add_argument("--processes", help="Number of Soap servers to run for parallel parsing (default: 1)",
                        default=1, type=int)
    parser.add_argument("--include", help="Only include files whose basenames are listed in the given text file")
    parser.add_argument("--log", help="Output a log of files processed to a file")
    parser.add_argument("--progress", help="Show a progress bar (should only be used if --log is given)",
                        action="store_true")
    parser.add_argument("--candc-args", nargs=argparse.REMAINDER, help="arguments to pass on to the C&C server (all "
                        "remaining arguments will be passed on)")
    opts = parser.parse_args()

    print >>sys.stderr, "#### C&C batch parser ####"
    ports = list(range(opts.port, opts.port+opts.processes))
    print >>sys.stderr, "Running Soap server(s) on %s:[%s]" % (opts.host, ", ".join([str(p) for p in ports]))

    print "opts.input_dir",opts.input_dir

    if not os.path.exists(opts.input_dir):
        print >>sys.stderr, "Non-existent input directory: %s" % opts.input_dir
        sys.exit(1)

    # Load a list of filenames to include if one was given
    if opts.include is not None:
        with open(opts.include, 'r') as include_file:
            include_list = include_file.read().split("\n")
    else:
        include_list = None
    
    # Initialize logging
    if opts.log:
        # Send logging to a file
        log_filename = opts.log
    else:
        # Use a stream logger
        log_filename = None
    # Configure the root logger
    logging.basicConfig(filename=opts.log, level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%y-%m-%d %H:%M')
    # This is the logger we'll use
    logger = logging.getLogger("candc")
    
    try:
        # Parse the whole directory
        parse_directory(opts.input_dir,
                        opts.dep_dir, opts.pos_dir,
                        host=opts.host, ports=list(range(opts.port, opts.port+opts.processes)),
                        include=include_list,
                        extra_args=opts.candc_args,
                        show_progress=opts.progress,
                        pool_name=opts.pool_name)
    finally:
        logging.shutdown()
