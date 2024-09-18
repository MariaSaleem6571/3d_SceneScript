from three_d_scene_script.metrics import run_all_metrics

gt_image_path = '/home/mseleem/3d_SceneScript/images/model_view_bottom.png'
output_image_path = '/home/mseleem/3d_SceneScript/images/model_view_side1.png'
results = run_all_metrics(gt_image_path, output_image_path)
print(results)
