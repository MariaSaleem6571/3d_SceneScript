import os
import json
import numpy as np
import matplotlib.pyplot as plt
from three_d_scene_script.metrics import run_all_metrics

def process_all_scenes(gt_images_dir, generated_images_dir, output_metrics_file):
    """Process all scenes and compute metrics."""
    gt_scene_map = {}
    for folder_name in os.listdir(gt_images_dir):
        if folder_name.endswith('_images'):
            scene_id = folder_name[:-len('_images')]
            gt_scene_map[scene_id] = folder_name

    gen_scene_map = {}
    for folder_name in os.listdir(generated_images_dir):
        if folder_name.startswith('generated_script_') and folder_name.endswith('_images'):
            scene_id = folder_name[len('generated_script_'):-len('_images')]
            gen_scene_map[scene_id] = folder_name
    scene_ids = sorted(set(gt_scene_map.keys()) & set(gen_scene_map.keys()))

    if not scene_ids:
        print("No matching scene IDs found between ground truth and generated images.")
        return

    all_metrics = []
    per_scene_metrics = {}

    for scene_id in scene_ids:
        print(f"Processing scene {scene_id}")
        gt_folder_name = gt_scene_map[scene_id]
        gen_folder_name = gen_scene_map[scene_id]

        gt_scene_path = os.path.join(gt_images_dir, gt_folder_name)
        gen_scene_path = os.path.join(generated_images_dir, gen_folder_name)

        gt_images = sorted(os.listdir(gt_scene_path))
        gen_images = sorted(os.listdir(gen_scene_path))

        image_filenames = sorted(set(gt_images) & set(gen_images))

        if not image_filenames:
            print(f"No matching images found for scene {scene_id}")
            continue

        scene_metrics = []

        for image_name in image_filenames:
            gt_image_path = os.path.join(gt_scene_path, image_name)
            gen_image_path = os.path.join(gen_scene_path, image_name)
            results = run_all_metrics(gt_image_path, gen_image_path)
            scene_metrics.append(results)
            all_metrics.append(results)
        metrics_keys = scene_metrics[0].keys()
        median_metrics = {}
        for key in metrics_keys:
            median_metrics[key] = np.median([m[key] for m in scene_metrics])

        per_scene_metrics[scene_id] = median_metrics

    if not all_metrics:
        print("No metrics computed. Please check your data.")
        return
    metrics_keys = all_metrics[0].keys()
    all_metrics_array = {key: np.array([m[key] for m in all_metrics]) for key in metrics_keys}

    overall_stats = {}
    for key in metrics_keys:
        overall_stats[key] = {
            'mean': float(np.mean(all_metrics_array[key])),
            'median': float(np.median(all_metrics_array[key])),
            'std': float(np.std(all_metrics_array[key]))
        }
    results = {
        'per_scene_metrics': per_scene_metrics,
        'overall_stats': overall_stats
    }

    with open(output_metrics_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Metrics saved to {output_metrics_file}")
    plot_metrics_per_scene(per_scene_metrics, metrics_keys)
    print("\nFinal Overall Statistics:")
    for key in metrics_keys:
        print(f"{key}: Mean = {overall_stats[key]['mean']:.4f}, Median = {overall_stats[key]['median']:.4f}, Std = {overall_stats[key]['std']:.4f}")

def plot_metrics_per_scene(per_scene_metrics, metrics_keys):
    """Plot each metric per scene using line graphs."""
    scenes = sorted(per_scene_metrics.keys())
    num_scenes = len(scenes)
    scene_indices = list(range(1, num_scenes + 1))

    for key in metrics_keys:
        values = [per_scene_metrics[scene][key] for scene in scenes]
        plt.figure(figsize=(10, 6))
        plt.plot(scene_indices, values, marker='o', linestyle='-')
        if num_scenes > 20:
            tick_step = max(num_scenes // 10, 1)
            tick_positions = scene_indices[::tick_step]
            plt.xticks(tick_positions)
        else:
            plt.xticks(scene_indices)
        plt.xlabel('Scene Number')
        plt.ylabel(key)
        plt.title(f'{key} per Scene')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{key}_per_scene.png')
        plt.close()
        print(f'Plot saved: {key}_per_scene.png')

if __name__ == "__main__":
    gt_images_dir = '../scene_images_gt'
    generated_images_dir = '../scene_images_generated'
    output_metrics_file = '../metrics_results.json'

    process_all_scenes(gt_images_dir, generated_images_dir, output_metrics_file)
