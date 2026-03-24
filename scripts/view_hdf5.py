#!/usr/bin/env python3
import h5py
import cv2
import numpy as np
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Export HDF5 recording to mp4')
    parser.add_argument('--file', type=str, required=True, help='Path to .hdf5 file')
    parser.add_argument('--fps', type=int, default=30, help='Playback FPS')
    parser.add_argument('--out', type=str, default=None, help='Output mp4 path')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] File not found: {args.file}")
        sys.exit(1)

    if args.out is None:
        args.out = os.path.splitext(args.file)[0] + ".mp4"

    with h5py.File(args.file, 'r') as f:
        print(f"File: {args.file}")
        print(f"Keys: {list(f.keys())}")

        qpos = f['observations/qpos'][:]
        img_top_key = 'observations/images/cam_high' if 'observations/images/cam_high' in f else 'observations/images/cam_top'
        img_wrist_key = 'observations/images/cam_wrist'

        images_top = f[img_top_key][:] if img_top_key in f else None
        images_wrist = f[img_wrist_key][:] if img_wrist_key in f else None

        length = len(qpos)
        print(f"Total frames: {length}")
        print(f"Qpos shape: {qpos.shape}")
        if images_top is not None:
            print(f"Top img shape: {images_top.shape}")
        if images_wrist is not None:
            print(f"Wrist img shape: {images_wrist.shape}")

        if images_top is None and images_wrist is None:
            print("[!] No image data found")
            return

        if images_top is not None and images_wrist is not None:
            h = max(images_top.shape[1], images_wrist.shape[1])
            w = images_top.shape[2] + images_wrist.shape[2]
        elif images_top is not None:
            h, w = images_top.shape[1], images_top.shape[2]
        else:
            h, w = images_wrist.shape[1], images_wrist.shape[2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

        for i in range(length):
            disp_img = None

            if images_top is not None:
                img_t = images_top[i].copy()
                disp_img = img_t

            if images_wrist is not None:
                img_w = images_wrist[i].copy()
                if disp_img is None:
                    disp_img = img_w
                else:
                    if disp_img.shape[0] != img_w.shape[0]:
                        img_w = cv2.resize(img_w, (img_w.shape[1], disp_img.shape[0]))
                    disp_img = np.hstack([disp_img, img_w])

            info_txt = f"Frame: {i}/{length}  Qpos: {qpos[i][:6].round(3)}"
            cv2.putText(disp_img, info_txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            writer.write(disp_img)

        writer.release()
        print(f"[+] Saved video to: {args.out}")

if __name__ == '__main__':
    main()