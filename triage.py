import os
import pickle
import random
import cv2
import numpy as np


def visualize_image_and_graph(img, nodes, edges, config,viz_img_size=512,pre=True):
    # img is rgb
    # Node coordinates in [0, 1], representing the normalized (r, c)
    # (r, c) -> (x, y)
    nodes = nodes[:, ::-1]

    # Resize the image to the specified visualization size, RGB->BGR
    img = cv2.resize(img, (viz_img_size, viz_img_size))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if config.DATA_FUSION_TYPE in ['ITO','IO','IT','I']:
    #     img = cv2.cvtColor(img[:,:,0:3], cv2.COLOR_RGB2BGR)
    # elif config.DATA_FUSION_TYPE in ['TO']:
    #     img = img[:, :, 0].astype(np.uint8)
    # else:#['T']
    #     img=np.squeeze(img).astype(np.uint8)

    # Draw edges
    for edge in edges:
        start_node = nodes[edge[0]] * viz_img_size
        end_node = nodes[edge[1]] * viz_img_size
        a=int(start_node[0])
        b=(int(start_node[0]), int(start_node[1]))
        c=(int(end_node[0]), int(end_node[1]))
        if pre==True:
            cv2.line(
                img,
                (int(start_node[0]), int(start_node[1])),
                (int(end_node[0]), int(end_node[1])),
                (15, 160, 253),
                5,)#orignal is 4

        else:
            cv2.line(
                img,
                (int(start_node[0]), int(start_node[1])),
                (int(end_node[0]), int(end_node[1])),
                (0, 0, 255),
                5,)#orignal is 4

    # Draw nodes
    for node in nodes:
        x, y = node * viz_img_size
        if pre==True:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)#orignal is 4
        else:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)#orignal is 4


    """
    for edge in edges:
        start_node = nodes[edge[0]] * viz_img_size
        end_node = nodes[edge[1]] * viz_img_size
        a=int(start_node[0])
        b=(int(start_node[0]), int(start_node[1]))
        c=(int(end_node[0]), int(end_node[1]))
        if pre==True:
            if config.DATA_FUSION_TYPE in ['ITO', 'IO', 'IT', 'I']:
                cv2.line(
                    img,
                    (int(start_node[0]), int(start_node[1])),
                    (int(end_node[0]), int(end_node[1])),
                    (15, 160, 253),
                    4,)
            else:  # ['TO','T']
                img1=np.zeros((1024,1024),dtype=np.uint8)
                cv2.line(
                    img,
                    (int(start_node[0]), int(start_node[1])),
                    (int(end_node[0]), int(end_node[1])),
                    255,
                    4,)
        else:
            if config.DATA_FUSION_TYPE in ['ITO', 'IO', 'IT', 'I']:
                cv2.line(
                    img,
                    (int(start_node[0]), int(start_node[1])),
                    (int(end_node[0]), int(end_node[1])),
                    (0, 0, 255),
                    4,)
            else:  # ['TO','T']
                cv2.line(
                    img,
                    (int(start_node[0]), int(start_node[1])),
                    (int(end_node[0]), int(end_node[1])),
                    (255),
                    4,)
    # Draw nodes
    for node in nodes:
        x, y = node * viz_img_size
        if pre==True:
            if config.DATA_FUSION_TYPE in ['ITO', 'IO', 'IT', 'I']:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)
            else:  # ['TO','T']
                cv2.circle(img, (int(x), int(y)), 4, (255), -1)
        else:
            if config.DATA_FUSION_TYPE in ['ITO', 'IO', 'IT', 'I']:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
            else:  # ['TO','T']
                cv2.circle(img, (int(x), int(y)), 4, (255), -1)
    """
    return img


def geo_metrics_image(img, nodes, edges,pre=True):
    # img is rgb
    # Node coordinates in [0, 1], representing the normalized (r, c)
    # (y, x) -> (x, y)
    nodes = nodes[:, ::-1]
    H,W=img.shape[0],img.shape[1]
    mask=np.zeros((H,W),dtype=np.int8)
    img_size=img.shape[0]

    # Draw edges
    for edge in edges:
        start_node = nodes[edge[0]] * img_size
        end_node = nodes[edge[1]] * img_size
        if pre==True:
            cv2.line(
                mask,
                (int(start_node[0]), int(start_node[1])),
                (int(end_node[0]), int(end_node[1])),
                (255),
                1,
            )
        else:
            cv2.line(
                mask,
                (int(start_node[0]), int(start_node[1])),
                (int(end_node[0]), int(end_node[1])),
                (255),
                1,
            )
    return mask


def rasterize_graph(nodes, edges, viz_img_size, dilation_radius):
    # Rasterize the graph.
    # Node coordinates in [0, 1], representing the normalized (r, c)

    # (r, c) -> (x, y)
    nodes = nodes[:, ::-1]

    # Creates the canvas
    img = np.zeros((viz_img_size, viz_img_size, 3), dtype=np.uint8)

    # Draw predicted nodes as white squares
    for node in nodes:
        x, y = node * viz_img_size
        cv2.rectangle(
            img,
            (int(x) - dilation_radius, int(y) - dilation_radius),
            (int(x) + dilation_radius, int(y) + dilation_radius),
            (255, 255, 255),
            -1,
        )

    # Draw predicted edges as white lines
    for edge in edges:
        start_node = nodes[edge[0]] * viz_img_size
        end_node = nodes[edge[1]] * viz_img_size
        cv2.line(
            img,
            (int(start_node[0]), int(start_node[1])),
            (int(end_node[0]), int(end_node[1])),
            (255, 255, 255),
            dilation_radius * 2,
        )

    return img


def visualize_pred_gt_pair(result):
    img = cv2.imread(result["img_path"])
    pred_img = visualize_image_and_graph(
        img, result["pred_nodes"], result["pred_edges"]
    )
    gt_img = visualize_image_and_graph(img, result["gt_nodes"], result["gt_edges"])
    pair_img = np.concatenate((pred_img, gt_img), axis=1)
    return pair_img


if __name__ == '__main__':
    # Deserializing the list from the binary file
    with open("inference_results.pickle", "rb") as file:
        inference_results = pickle.load(file)

    result_length = len(inference_results)
    worst_ratio = 0.1
    sample_num = 200

    output_dir = "triage/below_average"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sorted_results = sorted(inference_results, key=lambda x: -x["smd"])

    # selected_results = sorted_results[:int(result_length * worst_ratio)]
    selected_results = [x for x in inference_results if x["smd"] > 0.05]

    sampled_results = random.sample(selected_results, sample_num)

    sampled_results = sorted(sampled_results, key=lambda x: -x["smd"])

    for x in sampled_results:
        pair_img = visualize_pred_gt_pair(x)
        smd = x["smd"]
        img_name = os.path.basename(x["img_path"])
        output_name = f"smd_{smd:.6f}_{img_name}"
        cv2.imwrite(os.path.join(output_dir, output_name), pair_img)
