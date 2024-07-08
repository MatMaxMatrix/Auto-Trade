# git_push.py
import subprocess

def git_push():
    # Add changes to git
    subprocess.run(['git', 'add', 'batch_data.json'])

    # Commit changes
    subprocess.run(['git', 'commit', '-m', 'Update batch_data.json'])

    # Push changes
    try:
        subprocess.run(['git', 'push'], check=True)
    except subprocess.CalledProcessError:
        # If push fails because the upstream branch is not set, set it and try again
        subprocess.run(['git', 'push', '--set-upstream', 'Auto-Trade', 'main'])
        subprocess.run(['git', 'push'])

if __name__ == "__main__":
    git_push()
