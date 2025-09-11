import prettytable as pt
import logging

def setup_logging(log_path):
    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def get_logger(args, parser, log_path):
    setup_logging(log_path)
    
    tb = pt.PrettyTable()
    tb.field_names = ["Argument", "Value", "Description"]
    tb.align["Argument"] = "l"
    tb.align["Value"] = "l"
    tb.align["Description"] = "l"

    for action in parser._actions:
        if action.dest == 'help' or action.dest == 'h':
            continue
        name = action.dest
        value = getattr(args, name, None)
        help_text = action.help if hasattr(action, 'help') and action.help else ''
        
        if value is None and action.default is not None:
            value = action.default
        if value is None:
            value = "None"
        
        tb.add_row([name, str(value), help_text])

    logger = logging.getLogger(__name__)
    logger.info("Experiment Arguments:\n%s", tb.get_string())
    return logger