import sys
import logging
import traceback

def error_message_detail(error: Exception, error_detail: sys) -> str:
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        last = traceback.extract_stack()[-2]
        file_name = last.filename
        line_no = last.lineno
    else:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno

    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_no}] error message [{str(error)}]"
    )

class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.exception("Divide by zero error")
        raise CustomException(e, sys) from e