import numpy as np
import matplotlib.pyplot as plt
import os
import math as m
import cv2 # type: ignore

# 壁から離れようとする処理
def v_wall(Lx,Ly,x,y):

    # parameters for repulsive force
    H = 1
    a = 5

    # 壁によって生じるx,y方向の速度
    vx_wall = H * m.cosh(a*x)**-2 - H * m.cosh(a*(x-Lx))**-2
    vy_wall = H * m.cosh(a*y)**-2 - H * m.cosh(a*(y-Ly))**-2

    return [vx_wall,vy_wall]


# 群れっぽい動きを再現する関数
def update_direction(Lx,Ly,x,y,theta,cutoff):
    # 動きの規則
    # 1. みんなと同じ方向に動く
    # 2. みんなと近づきすぎると離れる方向に動く
    # 3. みんなと離れすぎると近づく方向に動く

    for i in range(0,len(x)):

        theta_list = []    

        for j in range(0,len(x)):
            
            # 自分と異なるデータ点のみ考慮
            if i == j: continue

            rel_pos = [x[j] - x[i], y[j] - y[i]]
            
            if(np.linalg.norm(rel_pos) < cutoff):
                theta_list.append(theta[j])


        # update velocity
        # 周囲の速度の平均
        new_theta = theta[i]
        vx_points = vy_points = 0

        # 配列が空でないときのみ平均を求める
        if theta_list:
            
            vx_points = np.average(np.cos(theta_list))
            vy_points = np.average(np.sin(theta_list))

        # 壁からの反発で生じる速度
        vx_wall,vy_wall = v_wall(Lx,Ly,x[i],y[i])
        # print(vx_wall)
        # print(x[i],vx_wall, vy_wall)

        new_theta = np.arctan2(vy_points + vy_wall, vx_points + vx_wall)

        # 周囲の方向を用いて自分の方向を修正
        alpha = 0.5

        theta[i] = alpha * theta[i] + (1-alpha) * new_theta

    # print("---")



# 壁からの距離に応じた反発力を速度として受ける


# 時間ステップ数
num_steps = 500
# 各時間ステップで生成する点の数
num_points = 50

# x, y の範囲
Lx = 10  # x座標の範囲
Ly = 10   # y座標の範囲
dt = 0.1

speed = 1
cutoff = 1 # この半径内にいるデータの情報を利用する

# 動画のfps（フレームレート）
fps = 30

# スクリプトが存在するディレクトリに画像を保存
output_dir = os.path.dirname(os.path.abspath(__file__))

# 画像保存用リスト
image_files = []

xs = np.random.rand(num_points) * Lx  # 0からLxの範囲でランダムなx座標
ys = np.random.rand(num_points) * Ly  # 0からLyの範囲でランダムなy座標
thetas = 2 * np.pi * np.random.rand(num_points)

# xs = [1]  # 0からLxの範囲でランダムなx座標
# ys = [9]  # 0からLyの範囲でランダムなy座標
# thetas = [np.pi / 2]

# print(thetas)

# vx = speed * np.cos(theta) # 0からLxの範囲でランダムなx座標
# vy = speed * np.sin(theta)   # 0からLyの範囲でランダムなy座標

for i in range(1, num_steps + 1):
    # ランダムな二次元座標 (xj, yj) を生成
    
    # プロット
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.gca().set_aspect('equal')  # アスペクト比を固定
    plt.title(f"Time step {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # プロットを画像ファイルに保存
    filename = f"{i:06d}.png"  # 右詰め6桁のファイル名
    filepath = os.path.join(output_dir, filename)  # スクリプトのディレクトリに保存
    plt.savefig(filepath)
    image_files.append(filepath)

    # 表示を閉じる
    plt.close()

    # update velocity
    update_direction(Lx,Ly,xs,ys,thetas,cutoff)

    xs = xs + speed * np.cos(thetas) * dt
    ys = ys + speed * np.sin(thetas) * dt

    print(f"Time step {i} のプロットを {filepath} に保存しました。")
    # print(thetas)

# 画像から動画を作成
video_filename = os.path.join(output_dir, "swarm.mp4")

# 画像のサイズを取得
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape
size = (width, height)

# 動画作成
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for filename in image_files:
    img = cv2.imread(filename)
    out.write(img)

out.release()

print(f"動画 {video_filename} が作成されました。")
