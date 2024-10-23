import logging

def setup_logger(log_file=None, level=logging.INFO):
    """
    Set up a logger for tracking training and evaluation progress.

    This function configures a logger to output log messages to both the console and an optional log file.
    The logger can be customized by specifying the logging level (e.g., INFO, DEBUG) and an optional log file.

    Parameters:
    log_file (str, optional): The file path where log messages will be saved. If None, logs are only printed to the console.
    level (int): The logging level, such as logging.INFO or logging.DEBUG.

    Returns:
    logging.Logger: Configured logger instance for use throughout the training and evaluation pipeline.
    """
    logger = logging.getLogger("ModelLogger")
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Check if the logger already has handlers to avoid duplicating log messages
    if not logger.hasHandlers():
        # Console handler for outputting logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler for saving logs to a specified file (if provided)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    # Example usage for setting up the logger
    try:
        logger = setup_logger("training.log", level=logging.INFO)
        logger.info("Training started")
        logger.debug("Debugging information for detailed analysis.")
    except Exception as e:
        print(f"An error occurred while setting up the logger: {e}")
