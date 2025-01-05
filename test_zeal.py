from zeal import ZEAL
import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image  
import numpy as np
def test_queries():
    zeal = ZEAL(device="cuda:0")  # Use "cpu" if you don't have GPU
    
    # Test single action
    action = "CliffDiving"
    queries = zeal.generate_queries(action)
    print(f"\nQueries for {action}:")
    print(f"Start Query: {queries['start_query']}")
    print(f"End Query: {queries['end_query']}")
    print(f"Description: {queries['description']}")

def test_action_filtering():
    zeal = ZEAL(device="cuda:0")
    
    # Example action classes from THUMOS14
    action_classes = [
        "BaseballPitch", "BasketballDunk", "Billiards", "CleanAndJerk",
        "CliffDiving", "CricketBowling", "CricketShot", "Diving",
        "FrisbeeCatch", "GolfSwing", "HammerThrow", "HighJump",
        "JavelinThrow", "LongJump", "PoleVault", "Shotput",
        "SoccerPenalty", "TennisSwing", "ThrowDiscus", "VolleyballSpiking"
    ]
    
    video_path = "/eph/nvme0/azureml/cr/j/3edb1ae0f02141f099c07ee15c43efe8/exe/wd/zeal/v_Basketball_g01_c04.mp4"
    relevant_actions = zeal.filter_relevant_actions(video_path, action_classes)
    print(f"Most relevant actions: {relevant_actions}")
def visualize_actionness_scores( scores: torch.Tensor, frames: List[Image.Image], action_name: str, save_path: Optional[str] = None):
    """
    Visualize actionness scores with histograms and key frames
    
    Args:
        scores (torch.Tensor): Actionness scores for each frame
        frames (List[Image.Image]): List of video frames
        action_name (str): Name of the action
        save_path (str, optional): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Convert scores to numpy
    scores_np = scores.cpu().numpy()
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    
    # Plot scores histogram
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(range(len(scores_np)), scores_np, color='blue', alpha=0.6)
    ax1.set_title(f'Actionness Scores Distribution for "{action_name}"')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Score')
    
    # Add horizontal grid lines
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Find top-k frames (e.g., top 3)
    k = 3
    top_k_indices = np.argsort(scores_np)[-k:]
    
    # Highlight top-k frames
    ax1.bar(top_k_indices, scores_np[top_k_indices], color='red', alpha=0.6)
    
    # Plot key frames
    ax2 = fig.add_subplot(gs[1])
    for idx, frame_idx in enumerate(top_k_indices):
        # Add frame as subplot
        frame = frames[frame_idx]
        ax_frame = fig.add_subplot(2, k, k+idx+1)
        ax_frame.imshow(frame)
        ax_frame.axis('off')
        ax_frame.set_title(f'Frame {frame_idx}\nScore: {scores_np[frame_idx]:.3f}')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_actionness_with_viz():
    zeal = ZEAL()
    
    # Get queries for an action
    action_name = "BasketballDunk"
    queries = zeal.generate_queries(action_name)
    
    # Load and process video
    video_path = "/eph/nvme0/azureml/cr/j/3edb1ae0f02141f099c07ee15c43efe8/exe/wd/zeal/v_Basketball_g01_c04.mp4"
    frames = zeal._extract_frames(video_path)
    
    # Compute actionness scores
    scores = zeal.compute_actionness_scores(frames, queries['description'])
    print(scores, queries['description'])
    # Visualize scores
    visualize_actionness_scores(
        scores=scores,
        frames=frames,
        action_name=action_name,
        save_path="actionness_viz.png"  # Optional: save to file
    )
def test_actionness():
    zeal = ZEAL()
    
    # First get queries for an action
    queries = zeal.generate_queries("BasketballDunk")
    
    # Load and process video
    video_path = "/eph/nvme0/azureml/cr/j/3edb1ae0f02141f099c07ee15c43efe8/exe/wd/zeal/v_Basketball_g01_c04.mp4"
    frames = zeal._extract_frames(video_path)
    
    # Compute actionness scores
    scores = zeal.compute_actionness_scores(frames, queries['description'])
    
    # Print scores
    print("\nActionness scores for each frame:")
    for i, score in enumerate(scores.cpu().numpy()):
        print(f"Frame {i}: {score:.4f}")
    
    # Get normalized scores
    norm_scores = zeal.normalize_scores(scores)
    print("\nNormalized scores:")
    for i, score in enumerate(norm_scores.cpu().numpy()):
        print(f"Frame {i}: {score:.4f}")

def test_lvlm_localization():
    zeal = ZEAL()
    
    # Get queries for an action
    action_name = "BasketballDunk"
    queries = zeal.generate_queries(action_name)
    
    # Load video frames
    video_path = "/eph/nvme0/azureml/cr/j/3edb1ae0f02141f099c07ee15c43efe8/exe/wd/zeal/v_Basketball_g01_c04.mp4"
    frames = zeal._extract_frames(video_path)
    
    # Process frames with LVLM
    start_scores, end_scores = zeal.process_frames_with_lvlm(
        frames,
        queries['start_query'],
        queries['end_query']
    )
    
    # Visualize the scores
    zeal.visualize_lvlm_scores(
        frames,
        start_scores,
        end_scores,
        action_name,
        save_path="lvlm_scores.png"
    )

def zeal_inferance(action_name, video_path):{

}
def  zeal_inferance(action_name, video_path):


  
    # Stage 1: Generate queries for an action
    #action_name = "BasketballDunk"
    queries = zeal.generate_queries(action_name)
    #print("\nGenerated Queries:")
    #print(f"Start Query: {queries['start_query']}")
    #print(f"End Query: {queries['end_query']}")
    #print(f"Description: {queries['description']}")

    # Load video and extract frames
    #video_path = "/eph/nvme0/azureml/cr/j/3edb1ae0f02141f099c07ee15c43efe8/exe/wd/zeal/v_Basketball_g01_c04.mp4"
    frames = zeal._extract_frames(video_path)
    #print(f"\nExtracted {len(frames)} frames from video")

    # Stage 2: Filter relevant actions (optional in this case since we know the action)
    action_classes = ["BasketballDunk", "BaseballPitch", "Diving", "CliffDiving", "football yellow card"]
    relevant_actions = zeal.filter_relevant_actions(video_path, action_classes)
    print(f"\nRelevant actions: {relevant_actions}")

    # Stage 3: Compute actionness scores
    actionness_scores = zeal.compute_actionness_scores(frames, queries['description'])
    #print(f"\nComputed actionness scores for {len(frames)} frames")

    # Stage 4: Get LVLM confidence scores for start/end
    start_scores, end_scores = zeal.process_frames_with_lvlm(
        frames,
        queries['start_query'],
        queries['end_query']
    )
    #print("\nComputed LVLM confidence scores")

    # Stage 5: Create interval proposals
    intervals = zeal.create_interval_proposals(
        start_scores=start_scores,
        end_scores=end_scores,
        actionness_scores=actionness_scores
    )
    print("\n",intervals[0]["start_time"],intervals[0]["end_time"],intervals[0]["conf_score"],intervals[0]['actionness_score'], intervals[0]["total_score"])
    # Print results
    #print("\nDetected Action Intervals:")
    #print(f"Action: {action_name}")
    '''
    for i, interval in enumerate(intervals, 1):
        print(f"\nInterval {i}:")
        print(f"Start Time: {interval['start_time']:.2f}")
        print(f"End Time: {interval['end_time']:.2f}")
        print(f"Confidence Score: {interval['conf_score']:.4f}")
        print(f"Actionness Score: {interval['actionness_score']:.4f}")
        print(f"Total Score: {interval['total_score']:.4f}")
    '''
if __name__ == "__main__":
    #test_queries()
    #test_action_filtering()    
    #test_actionness_with_viz()
    #test_actionness()
    #test_lvlm_localization()
    #main()    
    # Initialize ZEAL
    zeal = ZEAL(device="cuda:0")  # Use "cpu" if no GPU
    zeal_inferance("football yellow card", "/eph/nvme0/azureml/cr/j/3edb1ae0f02141f099c07ee15c43efe8/exe/wd/zeal/2410000.mp4")
    

    
