import pandas as pd
import subprocess
import os

if __name__ == "__main__":
    data = pd.read_csv("output/csv/slowfast_info_cliplen1.csv")
    for i in range(10):
        newdata = data.copy()
        newdata['feature_path'] = newdata['feature_path'].apply(lambda x: x.replace("slowfast_features_cliplen1",f"slowfast_features_cliplen1_start0{i}"))
        path = f'output/csv/slowfast_info_cliplen1_start0{i}.csv'
        newdata.to_csv(path)
        env = os.environ.copy()
        env["PYTHONPATH"] = f'{env.get("PYTHONPATH","")}:.' 
        command=f"python extract_feature/extract.py --csv {path} --start_time 0.{i} --clip_len 1 TEST.CHECKPOINT_FILE_PATH checkpoint/SLOWFAST_8x8_R50.pkl"
        try:
            result=subprocess.run(command.split(' '),env=env, check=True,text=True, capture_output=True)
            print(result.stderr)
            print(result.stdout)
            print(result.returncode)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
 



        
