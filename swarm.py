import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math as m
import cv2 # type: ignore

# 壁から離れようとする処理
def v_wall(Lx,Ly,x,y):

    # parameters for repulsive force
    H = 5
    a = 3

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

    cutoff_rep = 0.5

    for i in range(0,len(x)):

        theta_list = []
        x_rep_list = [] # 周囲から離れる時の処理に利用 x_repulsive_list
        y_rep_list = []

        for j in range(0,len(x)):
            
            # 自分と異なるデータ点のみ考慮
            if i == j: continue

            rel_pos = [x[j] - x[i], y[j] - y[i]]

            # 周囲との距離を一定に保つ処理に使用する位置のリストを取得
            if(np.linalg.norm(rel_pos) < cutoff_rep):
                theta_list.append(theta[j])
                x_rep_list.append(x[j])
                y_rep_list.append(y[j])
            
            # 方向修正に使用する周囲の点の方向のリストを取得
            if(np.linalg.norm(rel_pos) < cutoff):
                theta_list.append(theta[j])


        # update velocity
        # 周囲の速度の平均
        new_theta = theta[i]

        # 1. 周囲の進む向きに合わせて進行方向を修正
        vx_points = vy_points = 0

        # 配列が空でないときのみ平均を求める
        if theta_list:
            vx_points = np.average(np.cos(theta_list))
            vy_points = np.average(np.sin(theta_list))


        # 2. 周囲と近づきすぎたら離れる
        vx_rep = vy_rep = 0
        a = 0.1
        if x_rep_list:
            for j in range(0,len(x_rep_list)):

                rel_pos = [x_rep_list[j] - x[i], y_rep_list[j] - y[i]]
                square_norm = rel_pos[0]**2 + rel_pos[1]**2

                vx_rep  = vx_rep - rel_pos[0] / square_norm
                vy_rep  = vy_rep - rel_pos[1] / square_norm

            vx_rep = a * vx_rep / len(x_rep_list)
            vy_rep = a * vy_rep / len(y_rep_list)

        # 3. 壁からの反発で生じる速度
        vx_wall,vy_wall = v_wall(Lx,Ly,x[i],y[i])
        # print(vx_wall)
        # print(x[i],vx_wall, vy_wall)

        vx_tot = vx_points + vx_rep + vx_wall
        vy_tot = vy_points + vy_rep + vy_wall

        new_theta = np.arctan2(vy_tot, vx_tot)
        # wall_theta = np.arctan2(vy_wall, vx_wall)

        # 周囲の方向を用いて自分の方向を修正
        alpha = 0.8

        # theta[i] = beta * (alpha * theta[i] + (1-alpha) * new_theta) + (1 - beta) * wall_theta
        theta[i] = alpha * theta[i] + (1-alpha) * new_theta

    # print("---")



# 壁からの距離に応じた反発力を速度として受ける


# 時間ステップ数
num_steps = 1000
# 各時間ステップで生成する点の数
num_points = 50

# x, y の範囲
Lx = 10  # x座標の範囲
Ly = 10   # y座標の範囲
dt = 0.05

speed = 1
cutoff = 0.5 # この半径内にいるデータの情報を利用する

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
    ax = plt.axes()

    # 点の描画
    plt.scatter(xs, ys)

    # カットオフ半径の描画
    # for j in range(0,len(xs)):
    #     c = patches.Circle(xy=(xs[j], ys[j]), radius=cutoff, fill=False, ec='r')
    #     ax.add_patch(c)


    # 画面の設定
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
