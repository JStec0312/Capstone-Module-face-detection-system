import sys
from commands.help import do_help
from commands.vector import do_vector
from commands.cos_sim import do_cos_sim
COMMANDS = {
    "--help":do_help,
    "--vector": do_vector,
    "--cos_sim":do_cos_sim
}
if len(sys.argv) < 2:
    raise("not enough arguments")

command = sys.argv[1]
args = {}
for arg in sys.argv[2:]:
    if '=' in arg:
        if '=' not in arg:
            key = arg
            value = True
        else:
            key, value = arg.split('=', 1)
            key = key.replace('-', '')
        
        args[key] = value
    else:
        raise ValueError('Invalid argument' + arg)
    


if command in COMMANDS:
    COMMANDS[command](args)
else:
    raise ValueError("Command Not Found")


