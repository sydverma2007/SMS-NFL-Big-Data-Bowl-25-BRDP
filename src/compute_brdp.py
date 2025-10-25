import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import compute_vector_change, moving_average, get_accel_vectors

# -------------------------------------
# CONFIG
# -------------------------------------
DATA_PATH = "../data/train/"
OUTPUT_PATH = "../data/brdp_results.csv"
FRAME_RATE = 10  # frames per second

# -------------------------------------
# MAIN BRDP COMPUTATION
# -------------------------------------
def compute_brdp_for_play(df_play, ball_land):
    """
    Compute Ball Reaction Delay Penalty (BRDP) for all defenders in one play.
    BRDP = frames between QB release and defender's directional acceleration change toward ball.
    """
    defenders = df_play[df_play["player_side"] == "Defense"]["nfl_id"].unique()
    results = []

    for d in defenders:
        df_def = df_play[df_play["nfl_id"] == d].copy()
        df_def = get_accel_vectors(df_def)

        # Approximate "ball release frame" (could refine later using ball velocity data)
        release_frame = df_def["frame_id"].min()

        # Vector from defender to ball landing spot
        ball_vec = np.array([
            ball_land["ball_land_x"] - df_def["x"].iloc[0],
            ball_land["ball_land_y"] - df_def["y"].iloc[0]
        ])

        # Compute angle difference between defender acceleration and ball vector per frame
        df_def["vec_to_ball"] = df_def.apply(
            lambda r: np.array([
                ball_land["ball_land_x"] - r["x"],
                ball_land["ball_land_y"] - r["y"]
            ]),
            axis=1
        )

        df_def["angle_diff"] = df_def.apply(
            lambda r: compute_vector_change(
                np.array([r["ax"], r["ay"]]),
                r["vec_to_ball"]
            ),
            axis=1
        )

        # Smooth noise and find first significant "reaction"
        smoothed = moving_average(df_def["angle_diff"], window=5)
        min_idx = smoothed.idxmin()
        reaction_frame = df_def.loc[min_idx, "frame_id"]

        brdp = max(0, reaction_frame - release_frame)

        results.append({
            "nfl_id": d,
            "reaction_delay_frames": brdp,
            "reaction_delay_seconds": brdp / FRAME_RATE
        })

    return pd.DataFrame(results)


# -------------------------------------
# MAIN LOOP OVER ALL FILES
# -------------------------------------
def main():
    all_results = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    print(f"Processing {len(files)} input files...\n")

    total_plays = 0
    missing_ball_data = 0

    for f in tqdm(files):
        df = pd.read_csv(os.path.join(DATA_PATH, f))
        plays = df.groupby(["game_id", "play_id"])

        for (gid, pid), df_play in plays:
            total_plays += 1

            # Safely extract or approximate ball landing position
            if "ball_land_x" in df_play.columns and "ball_land_y" in df_play.columns:
                ball_land = df_play[["ball_land_x", "ball_land_y"]].iloc[0].to_dict()
            else:
                missing_ball_data += 1
                # Fallback logic for missing ball_land columns
                if "player_role" in df_play.columns:
                    target_receivers = df_play[df_play["player_role"] == "Targeted Receiver"]
                    if not target_receivers.empty:
                        ball_land = {
                            "ball_land_x": target_receivers["x"].mean(),
                            "ball_land_y": target_receivers["y"].mean()
                        }
                    else:
                        ball_land = {
                            "ball_land_x": df_play["x"].mean(),
                            "ball_land_y": df_play["y"].mean()
                        }
                else:
                    # If player_role is missing entirely
                    ball_land = {
                        "ball_land_x": df_play["x"].mean(),
                        "ball_land_y": df_play["y"].mean()
                    }

            # Compute BRDP for this play
            try:
                results = compute_brdp_for_play(df_play, ball_land)
                results["game_id"] = gid
                results["play_id"] = pid
                results["source_file"] = f
                all_results.append(results)
            except Exception as e:
                print(f"⚠️ Skipping play {gid}-{pid} due to error: {e}")
                continue

    # Combine and save all results
    if all_results:
        brdp_df = pd.concat(all_results, ignore_index=True)
        brdp_df.to_csv(OUTPUT_PATH, index=False)
        print("\n-------------------------------------")
        print(f"Saved BRDP results to: {OUTPUT_PATH}")
        print(f"Total plays processed: {total_plays}")
        print(f"Plays missing ball landing data: {missing_ball_data}")
        print(f"Total defender entries: {len(brdp_df)}")
        print("-------------------------------------\n")
    else:
        print("No valid BRDP results generated — check your data input.")

if __name__ == "__main__":
    main()