import numpy as np
import matplotlib.pyplot as plt
import os
import math as m
import cv2 # type: ignore

# 群れっぽい動きを再現する関数
def update_velocity(positions, velocities, cutoff):
    # 動きの規則
    # 1. みんなと同じ方向に動く
    # 2. みんなと近づきすぎると離れる方向に動く
    # 3. みんなと離れすぎると近づく方向に動く

    for i in range(0,len(positions[0])):

        x_list = []
        y_list = []
        vx_list = []    
        vy_list = []

        for j in range(0,len(positions[0])):

            rel_pos = [positions[0][j] - positions[0][i], positions[1][j] - positions[1][i]]
            
            if(np.linalg.norm(rel_pos) < cutoff):

                x_list.append(positions[0][j])
                y_list.append(positions[1][j])
                vx_list.append(velocities[0][j])
                vy_list.append(velocities[1][j])

        # update velocity
        # 周囲の速度の平均
        

        new_vx1 = np.average(vx_list)
        new_vy1 = np.average(vx_list)

        theta = np.arctan2(new_vy1,new_vx1)

        # 周囲の速度にある程度影響を受ける
        alpha = 0.9

        velocities[0][i] = alpha * velocities[0][i] + (1-alpha) * speed * np.cos(theta)
        velocities[1][i] = alpha * velocities[1][i] + (1-alpha) * speed * np.sin(theta)

# 壁から離れようとする処理
def away_from_wall(Lx,Ly,positions, velocities):

    for i in range(0,len(positions[0])):
        # 物理的にはおかしいが，hyperbolic tangentにしてみる
        pass


# 壁からの距離に応じた反発力を速度として受ける


# 時間ステップ数
num_steps = 10
# 各時間ステップで生成する点の数
num_points = 10

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

x = np.random.rand(num_points) * Lx  # 0からLxの範囲でランダムなx座標
y = np.random.rand(num_points) * Ly  # 0からLyの範囲でランダムなy座標

theta = 2*np.pi * np.random.rand(num_points)
vx = speed * np.cos(theta) # 0からLxの範囲でランダムなx座標
vy = speed * np.sin(theta)   # 0からLyの範囲でランダムなy座標

for i in range(1, num_steps + 1):
    # ランダムな二次元座標 (xj, yj) を生成
    
    # プロット
    plt.figure()
    plt.scatter(x, y)
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
    update_velocity([x,y],[vx,vy],cutoff)

    x = x + vx*dt
    y = y + vy*dt

    print(f"Time step {i} のプロットを {filepath} に保存しました。")

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
