def do_help(args):
    print("Commands:")
    print(vector_message)
    print(help_message)














vector_message = """
  --vector -img:<path> [-model_name:<name>] [-detector_backend:<name>] [-enforce_detection:<true|false>]
"""
help_message = """
  do_cos_sim_2 img_path1=<path> img_path2=<path> [model_name=<name>] [detector_backend=<name>] [enforce_detection=<true|false>]
"""