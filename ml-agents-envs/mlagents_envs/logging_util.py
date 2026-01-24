import logging  # noqa I251
import sys
import re
from typing import List, Pattern

CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

_loggers = set()
_log_level = NOTSET
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEBUG_LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
LOG_FORMAT = "[%(levelname)s] %(message)s"

# Patterns for sensitive data that should be scrubbed from logs
_SCRUB_PATTERNS: List[Pattern] = [
    re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
    re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
    re.compile(r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
    re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
    re.compile(r'auth["\']?\s*[:=]\s*["\']?([^"\'\s]+)', re.IGNORECASE),
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    # Credit card patterns (simple check)
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    # IP addresses (private ranges)
    re.compile(r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
]

_SCRUB_REPLACEMENT = "***REDACTED***"


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name. The logger will use the log level
    specified by set_log_level()
    """
    logger = logging.getLogger(name=name)

    if _log_level == DEBUG:
        formatter = logging.Formatter(fmt=DEBUG_LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter(fmt=LOG_FORMAT)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # If we've already set the log level, make sure new loggers use it
    if _log_level != NOTSET:
        logger.setLevel(_log_level)

    # Keep track of this logger so that we can change the log level later
    _loggers.add(logger)
    return logger


def set_log_level(log_level: int) -> None:
    """
    Set the ML-Agents logging level. This will also configure the logging format (if it hasn't already been set).
    """
    global _log_level
    _log_level = log_level

    for logger in _loggers:
        logger.setLevel(log_level)

    if log_level == DEBUG:
        formatter = logging.Formatter(fmt=DEBUG_LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter(LOG_FORMAT)
    _set_formatter_for_all_loggers(formatter)


def _set_formatter_for_all_loggers(formatter: logging.Formatter) -> None:
    for logger in _loggers:
        for handler in logger.handlers[:]:
            handler.setFormatter(formatter)


def scrub_sensitive_data(message: str) -> str:
    """
    Scrub sensitive data from log messages.

    Removes patterns that look like:
    - Passwords, API keys, tokens, secrets
    - Email addresses
    - Credit card numbers
    - Private IP addresses

    Args:
        message: The log message to scrub

    Returns:
        The message with sensitive data replaced by ***REDACTED***
    """
    scrubbed = message
    for pattern in _SCRUB_PATTERNS:
        scrubbed = pattern.sub(_SCRUB_REPLACEMENT, scrubbed)
    return scrubbed


class ScrubberFilter(logging.Filter):
    """
    Logging filter that scrubs sensitive data from log records.

    Usage:
        logger = get_logger(__name__)
        for handler in logger.handlers:
            handler.addFilter(ScrubberFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub the message
        record.msg = scrub_sensitive_data(str(record.msg))

        # Scrub args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: scrub_sensitive_data(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    scrub_sensitive_data(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        return True


def enable_log_scrubbing() -> None:
    """
    Enable log scrubbing for all existing loggers.

    Call this function to add scrubbing filters to all handlers.
    Should be called early in application startup.
    """
    scrubber = ScrubberFilter()
    for logger in _loggers:
        for handler in logger.handlers:
            handler.addFilter(scrubber)
