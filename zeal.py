from typing import Dict, List, Tuple, Optional
import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, LlavaOnevisionForConditionalGeneration
from openai import OpenAI
import os
import json
import os
from PIL import Image  
os.environ['OPENAI_API_KEY'] = ''
class ZEAL:
    def __init__(self, device: str = "cuda:2"):
        self.device = device
        # Initialize CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize LLaVA
        self.llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False
        ).to(device)
        self.llava_processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_queries(self, action_name: str) -> Dict[str, str]:
        """
        Stage 1: Generate queries for action localization using LLM
        
        Args:
            action_name (str): Name of the action to analyze
            
        Returns:
            Dict with three keys:
                - 'start_query': Question about action start
                - 'end_query': Question about action end
                - 'description': Short description of action
        """
        messages = [
            {
                "role": "user",
                "content": f"""Here is an action: {action_name}
Please follow these steps to create a start question, end question, and description for the action:
1. Consider what is unique about the very beginning of the action that could be observed in a single video frame. Craft a yes/no question about that.
2. Consider what is unique about the very end of the action that could be observed in a single video frame. Craft a yes/no question about that. Make sure it does not overlap with the start question.
3. Write a very short description summarizing the key components of the action, without using adverbs. The description should differentiate the action from other actions.

Output your final answer in this JSON format:
{{
    "{action_name}": {{
        "start": "question",
        "end": "question",
        "description": "description"
    }}
}}

Make sure to follow the JSON formatting exactly. Only output the JSON."""
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1,
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            
            # Extract queries and description
            queries = parsed[action_name]
            
            return {
                'start_query': queries['start'],
                'end_query': queries['end'],
                'description': queries['description']
            }
            
        except Exception as e:
            print(f"Error generating queries: {e}")
            # Return default queries if LLM fails
            return {
                'start_query': f"Is the {action_name} starting?",
                'end_query': f"Is the {action_name} complete?",
                'description': f"A person performs {action_name}"
            }
    def _extract_frames(self, video_path: str, num_frames: int = 16) -> List[Image.Image]:
        """
        Extract uniformly sampled frames from video
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to extract (paper uses 16)
        
        Returns:
            List of PIL Image objects
        """
        import cv2
        from PIL import Image
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total frame: ", total_frames)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(total_frames) / fps
        print("duration ",duration)
    

        # Calculate frame indices to sample
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                frame = Image.fromarray(frame)
                frames.append(frame)
        
        cap.release()
        return frames

    def filter_relevant_actions(self, video_path: str, action_classes: List[str], top_k: int = 3) -> List[str]:
        """
        Stage 2: Filter most relevant actions using CLIP
        
        Args:
            video_path (str): Path to video file
            action_classes (List[str]): List of possible action classes
            top_k (int): Number of top actions to return
            
        Returns:
            List of top-k most relevant action classes
        """
        # Extract frames
        frames = self._extract_frames(video_path)
        
        try:
            # Process images and text with CLIP
            image_inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            text_inputs = self.clip_processor(text=action_classes, return_tensors="pt", padding=True).to(self.device)

            # Get features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Average scores across frames
                mean_scores = similarity.mean(dim=0)
                
                # Get top-k actions
                top_k_values, top_k_indices = mean_scores.topk(min(top_k, len(action_classes)))
                
                selected_actions = [action_classes[idx] for idx in top_k_indices.cpu().numpy()]
                
                print(f"Selected actions with scores:")
                for action, score in zip(selected_actions, top_k_values.cpu().numpy()):
                    print(f"{action}: {score:.2f}")
                    
                return selected_actions

        except Exception as e:
            print(f"Error in filtering actions: {e}")
            # If error occurs, return first top_k actions from the list
            return action_classes[:top_k]     


    def compute_actionness_scores(self, frames: List[Image.Image], action_description: str) -> torch.Tensor:
        """
        Stage 3: Compute CLIP-based actionness scores for frames
        
        Args:
            frames (List[Image.Image]): List of video frames as PIL Images
            action_description (str): Short description of the action
            
        Returns:
            torch.Tensor: Actionness scores for each frame
        """
        try:
            # Process frames and action description with CLIP
            image_inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            text_inputs = self.clip_processor(text=[action_description], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                # Get image and text features
                image_features = self.clip_model.get_image_features(**image_inputs)
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).squeeze()
                
                # Apply softmax to get probabilities
                actionness_scores = torch.nn.functional.softmax(similarity, dim=0)
                
                return actionness_scores

        except Exception as e:
            print(f"Error computing actionness scores: {e}")
            # Return uniform scores if error occurs
            return torch.ones(len(frames), device=self.device) / len(frames)

    def normalize_scores(self, scores: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        """
        Apply modified min-max normalization to scores
        Args:
            scores (torch.Tensor): Input scores
            epsilon (float): Small value to avoid extremes (0 or 1)
        Returns:
            torch.Tensor: Normalized scores
        """
        # Print input scores for debugging
        print("Before normalization:", scores)
        
        min_val = scores.min()
        max_val = scores.max()
        print(f"min_val: {min_val}, max_val: {max_val}")
        
        # Check if all values are the same
        if max_val == min_val:
            return torch.full_like(scores, 0.5)  # Return uniform scores
            
        # Apply min-max normalization with epsilon adjustment
        normalized = (scores - min_val) / (max_val - min_val)
        normalized = normalized * (1 - 2 * epsilon) + epsilon
        
        # Print output scores for debugging
        print("After normalization:", normalized)
        
        return normalized 
        
    def get_lvlm_confidence(self, frame: Image.Image, query: str, query_type: str) -> float:
        """
        Get LVLM confidence score following the paper's method
        Args:
            frame (Image.Image): Frame to analyze
            query (str): Question about the frame
            query_type (str): Either 'start' or 'end' to identify query type
        Returns:
            float: 1.0 for yes, 0.0 for no
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{query} Only answer yes or no."},
                ],
            },
        ]
        
        try:
            # Process input
            prompt = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.llava_processor(
                images=frame, 
                text=prompt, 
                return_tensors="pt"
            ).to(self.device, torch.float16)
            
            # Generate response
            with torch.no_grad():
                outputs = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    pad_token_id=self.llava_processor.tokenizer.pad_token_id,
                    eos_token_id=self.llava_processor.tokenizer.eos_token_id
                )
                
            # Decode and get response
            full_response = self.llava_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant's answer
            if 'assistant' in full_response.lower():
                response = full_response.lower().split('assistant')[-1].strip()
            else:
                response = full_response.lower().strip()
                
            print("\n" + "="*50)
            print(f"Query Type: {query_type.upper()}")
            print(f"Query: {query}")
            print(f"Full response: {full_response}")
            print(f"Extracted answer: {response}")
            
            # Simple binary scoring as per paper
            if 'yes' in response:
                confidence = 1.0
            elif 'no' in response:
                confidence = 0.0
            else:
                confidence = 0.5  # For unclear responses
                
            print(f"Computed confidence for {query_type}: {confidence}")
            print("="*50 + "\n")
            return confidence
            
        except Exception as e:
            print("\n" + "="*50)
            print(f"Error getting LVLM confidence for {query_type} query: {e}")
            print(f"Query was: {query}")
            print("="*50 + "\n")
            return 0.5
    def process_frames_with_lvlm(
        self, 
        frames: List[Image.Image], 
        start_query: str,
        end_query: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process all frames with LVLM to get start and end confidence scores
        """
        start_scores = []
        end_scores = []
        
        print("Processing frames with LVLM...")
        for i, frame in enumerate(frames):
            print(f"\nProcessing frame {i+1}/{len(frames)}")
            
            # Get confidence scores for start and end queries
            start_conf = self.get_lvlm_confidence(frame, start_query, "start")
            end_conf = self.get_lvlm_confidence(frame, end_query, "end")
            
            start_scores.append(start_conf)
            end_scores.append(end_conf)

        print("start score : ", start_scores)
        print("end score : ", end_scores)
        # Convert to tensors
        start_scores = torch.tensor(start_scores, device=self.device)
        end_scores = torch.tensor(end_scores, device=self.device)
        print("c start score : ", start_scores)
        print(" c end score : ", end_scores)
        # Normalize scores
        start_scores = self.normalize_scores(start_scores)
        end_scores = self.normalize_scores(end_scores)
        print("nstart score : ", start_scores)
        print("n end score : ", end_scores)
        return start_scores, end_scores

    def visualize_lvlm_scores(
        self,
        frames: List[Image.Image],
        start_scores: torch.Tensor,
        end_scores: torch.Tensor,
        action_name: str,
        save_path: Optional[str] = None
    ):
        """
        Visualize LVLM confidence scores for start and end queries
        """
        import matplotlib.pyplot as plt
        
        # Convert scores to numpy
        start_np = start_scores.cpu().numpy()
        end_np = end_scores.cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot scores
        plt.plot(start_np, 'b-', label='Start Confidence', alpha=0.7)
        plt.plot(end_np, 'r-', label='End Confidence', alpha=0.7)
        
        plt.title(f'LVLM Confidence Scores for {action_name}')
        plt.xlabel('Frame Index')
        plt.ylabel('Confidence Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()       
    def create_interval_proposals(
        self,
        start_scores: torch.Tensor,
        end_scores: torch.Tensor,
        actionness_scores: torch.Tensor,
        threshold: float = 0.5,
        lambda_weight: float = 0.3  # As mentioned in paper
    ) -> List[Dict[str, float]]:
        """
        Create action interval proposals from confidence scores
        
        Args:
            start_scores: Confidence scores for action start
            end_scores: Confidence scores for action end
            actionness_scores: CLIP-based actionness scores
            threshold: Threshold for considering high confidence frames
            lambda_weight: Weight for combining confidence and actionness scores
            
        Returns:
            List of intervals with start_time, end_time, and confidence scores
        """
        # Get candidate start and end frames
        print("start_scores: ", start_scores)
        print("end_scores: " ,end_scores)
        start_candidates = torch.nonzero(start_scores >= threshold).squeeze(1)
        end_candidates = torch.nonzero(end_scores >= threshold).squeeze(1)
        
        intervals = []
        
        print( "Create intervals from all possible start-end pairs")
        for start_idx in start_candidates:
            for end_idx in end_candidates:
                # Only consider valid intervals where end comes after start
                if end_idx <= start_idx:
                    continue
                    
                print("Calculate boundary confidence")
                conf_score = (start_scores[start_idx] + end_scores[end_idx]) / 2
                
                # Calculate actionness score for the interval
                interval_actionness = actionness_scores[start_idx:end_idx+1].mean()
                
                # Combine scores as per paper
                total_score = (lambda_weight * conf_score + 
                            (1 - lambda_weight) * interval_actionness)
                print("total_score",total_score)
                intervals.append({
                    'start_frame': int(start_idx),
                    'end_frame': int(end_idx),
                    'start_time': float(start_idx),  # Convert to seconds if needed
                    'end_time': float(end_idx),
                    'conf_score': float(conf_score),
                    'actionness_score': float(interval_actionness),
                    'total_score': float(total_score)
                })
        
        # Sort intervals by score
        intervals = sorted(intervals, key=lambda x: x['total_score'], reverse=True)
        
        return self.non_maximum_suppression(intervals)

    def non_maximum_suppression(
        self,
        intervals: List[Dict[str, float]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Apply Non-Maximum Suppression to remove overlapping intervals
        
        Args:
            intervals: List of interval dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of intervals
        """
        if not intervals:
            return []
            
        # Sort by score
        intervals = sorted(intervals, key=lambda x: x['total_score'], reverse=True)
        selected = []
        
        while intervals:
            # Take the interval with highest score
            current = intervals.pop(0)
            selected.append(current)
            
            # Filter out overlapping intervals
            non_overlapping = []
            for interval in intervals:
                iou = self.compute_iou(current, interval)
                if iou < iou_threshold:
                    non_overlapping.append(interval)
            
            intervals = non_overlapping
        
        return selected

    def compute_iou(self, interval1: Dict[str, float], interval2: Dict[str, float]) -> float:
        """Compute IoU between two intervals"""
        start1, end1 = interval1['start_time'], interval1['end_time']
        start2, end2 = interval2['start_time'], interval2['end_time']
        
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = (end1 - start1) + (end2 - start2) - intersection
        
        return intersection / union if union > 0 else 0.0     
    def detect_action_intervals(self, video_path: str, action_name: str, top_k: int = 3) -> List[Dict[str, float]]:
        """
        Main function to detect action intervals in a video
        
        Args:
            video_path (str): Path to the video file
            action_name (str): Name of the action to detect
            top_k (int): Number of top intervals to return
            
        Returns:
            List[Dict[str, float]]: List of top-k intervals with start_time, end_time, and confidence
        """
        try:
            # Stage 1: Generate queries
            queries = self.generate_queries(action_name)
            
            # Extract frames from video
            frames = self._extract_frames(video_path)
            if not frames:
                raise ValueError("No frames extracted from video")
                
            # Stage 3: Compute actionness scores
            actionness_scores = self.compute_actionness_scores(frames, queries['description'])
            
            # Stage 4: Get LVLM confidence scores
            start_scores, end_scores = self.process_frames_with_lvlm(
                frames,
                queries['start_query'],
                queries['end_query']
            )
            
            # Stage 5: Create interval proposals
            intervals = self.create_interval_proposals(
                start_scores=start_scores,
                end_scores=end_scores,
                actionness_scores=actionness_scores
            )
            
            # Get video FPS for time conversion
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Convert frame indices to timestamps and format result
            formatted_intervals = []
            for interval in intervals[:top_k]:  # Take top-k intervals
                formatted_intervals.append({
                    'start_time': float(interval['start_frame']) / fps,
                    'end_time': float(interval['end_frame']) / fps,
                    'confidence': float(interval['total_score'])
                })
                
            return formatted_intervals
            
        except Exception as e:
            print(f"Error in detect_action_intervals: {e}")
            return []    
