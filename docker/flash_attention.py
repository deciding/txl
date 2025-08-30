from modal import Image, App

app = App(name="txl")  # Note: this is optional since Modal 0.57


txl_image = Image.from_dockerfile(path="./Dockerfile")

# Example function that uses the image
@app.function(gpu="H100", image=txl_image)
def test_flash_attention():
    def get_gpu_type():
        import subprocess
        try:
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if 'H100' in line and 'HBM3' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return False

    if not get_gpu_type():
        return
