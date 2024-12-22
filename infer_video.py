import os
import struct
import argparse
import cv2
import numpy as np
import onnxruntime
from timeit import default_timer as timer
from cvglue import warp_back, crop_face_v3

def parse_args():
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument('--model_path', type=str, help='model path *.onnx')
    parser.add_argument('--video_path', type=str, help='video path to inference')
    parser.add_argument('--lands_path', type=str, help='annotation path to inference')
    args = parser.parse_args()
    return args

def infer(model, frame, landmarks, snapshot=False, **kwargs):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1- 仿射变换
    kp = landmarks[[96,97,76,82]].flatten().tolist()
    lands = landmarks[:33]
    # kp = landmarks[[1, 0, 21, 15]].flatten().tolist()
    # lands = landmarks[-33:][::-1]
    net_shape = model.get_inputs()[0].shape
    src_img, warp_mat = crop_face_v3(rgb, kp, lands, output_size=(net_shape[2], net_shape[3]), **kwargs)

    # 2- 预处理
    input_x = (src_img - 127.5) * 0.007843137
    input_x = np.transpose(input_x, (2,0,1))
    input_x = np.float32(input_x[np.newaxis, :])

    # 3- 推理
    ort_inputs = {model.get_inputs()[0].name: input_x}
    ort_outputs = model.run(None, ort_inputs)

    # 4- 结果校对
    torch_out = np.load('local_torch_results.npy')
    np.testing.assert_allclose(ort_outputs[0], torch_out, rtol=1e-03, atol=1e-05)
    np.save('onnx_results.npy', ort_outputs[0])

    # 5- 后处理
    y_images = np.clip((ort_outputs[0] + 1)*127.5, 0.0, 255.0)
    y_images = np.transpose(y_images, (0,2,3,1))

    # 6- 贴回显示
    frame_out = warp_back(y_images[0], rgb, warp_mat)
    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

    # 7- 保存中间帧用于分析
    if snapshot == 10:
        np.save("./results/frame_in.npy", frame)
        np.save("./results/process_in.npy", src_img)
        np.save("./results/process_out.npy", y_images[0])
        cv2.imwrite('./results/local_'+str(frame_id)+'.png', cv2.cvtColor(y_images[0], cv2.COLOR_RGB2BGR))

    return frame_out


if __name__ == "__main__":
    opt = parse_args()
    os.makedirs('./results', exist_ok=True)

    # 1- 加载模型
    model = onnxruntime.InferenceSession(opt.model_path, providers=['CPUExecutionProvider'])

    net_shape = model.get_inputs()[0].shape
    antialias = False if net_shape[2] > 320 else True
    interp_mode = cv2.INTER_LINEAR

    # 预处理参数
    params = {'hp_factor': 1.7, 'wd_factor': 0.6, 'shift_factor': 1.1, 'interp_mode': interp_mode, 'antialias': antialias}

    start = timer()

    # 2- 处理视频
    capture = cv2.VideoCapture(opt.video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    # 样本人脸关键点
    lands_file = open(opt.lands_path, 'rb')
    lands_list = struct.unpack(frame_count*'196f', lands_file.read(frame_count*196*4))
    lands_array = np.reshape(lands_list, (frame_count, 98, 2))
    # lands_file = open(opt.lands_path, 'rb')
    # lands_list = struct.unpack(frame_count*'212f', lands_file.read(212*4*frame_count))
    # lands_array = np.reshape(lands_list, (frame_count, 106, 2))

    writer = cv2.VideoWriter(os.path.join('./results', opt.video_path.split('/')[-1]), cv2.VideoWriter_fourcc('M','P','4','V'), fps, (width, height))
    for frame_id in range(frame_count):
        ret, frame = capture.read()
        if ret:
            try:
                frame_out = infer(model, frame, lands_array[frame_id], snapshot=frame_id, **params)
            except:
                frame_out = frame
            writer.write(frame_out)
    capture.release()
    writer.release()

    end = timer()
    print("time elapsed: ", end-start, "s")
