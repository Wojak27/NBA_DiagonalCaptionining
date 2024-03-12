import subprocess
import os

def main(files_folder, output_folder):
    # get all mp4 files in the current folder
    files = [f for f in os.listdir(files_folder)]
    files = [f for f in files if f.endswith(".mp4")]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for f in files:
        # create the command to be executed
        file_name = f.split(".mp4")[0].split("/")[-1]
        file_name = file_name + ".wav"
        file_name = os.path.join(output_folder, file_name)
        command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(os.path.join(files_folder, f), file_name)
        # execute the command
        subprocess.call(command, shell=True)

if __name__ == "__main__":
    main("/home/karolwojtulewicz/code/TimeSformer/dataset/TH14_test_set_mp4", "/home/karolwojtulewicz/code/TimeSformer/dataset/TH14_test_set_wav")