import os
import subprocess
import sys


#verifies that pod really has GPU access 
def run_command(command):
    print(f"\n$ {' '.join(command)}")

    try:
        result=subprocess.run(command,check=False,text=True,capture_output=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    except FileNotFoundError:
        print(f"Command not found: {command[0]}")
        



def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())

    run_command(["nvidia-smi"])

    try:

        import torch 
        print("\nPyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("CUDA device count:", torch.cuda.device_count())
            print("Current CUDA device:", torch.cuda.current_device())
            print("GPU name:" ,torch.cuda.get_device_name(0))

    except Exception as e:
        print("Could not check PyTorch or Cuda:",repr(e))



if __name__ == "__main__":
    main()




