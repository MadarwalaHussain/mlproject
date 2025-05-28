# This module handles the exception handling for the data transformation process.

import sys
import logging
from src.logger import setup_logger, get_logger  # Import your custom logger setup


def error_message_details(error, error_detail: sys):
    """
    This function returns a detailed error message.

    Parameters:
    error (Exception): The exception that occurred.
    error_detail (sys): The sys module for accessing system-specific parameters and functions.

    Returns:
    str: A formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # gives the traceback of the error
    file_name = exc_tb.tb_frame.f_code.co_filename  # gets the name of the file where the error occurred
    error_message = f"Error occurred in script: {file_name}, line number: {exc_tb.tb_lineno}, error message: {str(error)}"
    return error_message


class CustomException(Exception):
    """
    Custom exception class for handling exceptions in the ML project.

    Inherits from the built-in Exception class and provides a custom error message.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with an error message.

        Parameters:
        error_message (Exception): The exception that occurred.
        error_detail (sys): The sys module for accessing system-specific parameters and functions.
        """
        super().__init__(error_message)  # Fixed: only pass the error_message
        # calls the function to get the error message
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        """
        Returns the string representation of the CustomException.

        Returns:
        str: The error message.
        """
        return self.error_message

   
