import os
import logging


def prepare_folder_for(filename_with_path):
    # Ensures foler exists for a given file path
    logging.info('preparing folder for "{}"'.format(filename_with_path))
    if not os.path.exists(os.path.dirname(filename_with_path)):
        try:
            os.makedirs(os.path.dirname(filename_with_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != os.errno.EEXIST:
                raise
