import docker
import time
import subprocess

# Initialize the Docker client
client = docker.from_env()

# Define the image to use and the command to run
image = 'python311:latest'
python_file_path = '/home/jaykchen/projects/autogen_rust/src/test.py'
container_python_file_path = '/test.py'

# Create the container with the Python file mounted
container = client.containers.run(
    image,
    'sleep infinity',  # Keep the container running
    detach=True,
    volumes={
        python_file_path: {'bind': container_python_file_path, 'mode': 'ro'}
    }
)
print(f"Container created with ID: {container.id}")

# Small delay to ensure the container is ready
time.sleep(1)

# Start the execution and capture the output
def main():
    try:
        # Use subprocess to handle stdin input
        process = subprocess.Popen(
            ['docker', 'exec', '-i', container.id, 'python', container_python_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Handle the output stream
        for chunk in iter(lambda: process.stdout.read(1), ''):
            print(chunk, end='')

        # Continuously send input to the process
        while True:
            user_input = input("Enter a number (or 'exit' to quit): ")
            process.stdin.write(f"{user_input}\n")
            process.stdin.flush()
            if user_input == 'exit':
                break

        # Wait for the process to complete
        process.wait()

    except docker.errors.NotFound as e:
        print(f"Error: {e}")
    finally:
        # Clean up: stop and remove the container
        container.stop()
        container.remove()

if __name__ == '__main__':
    main()