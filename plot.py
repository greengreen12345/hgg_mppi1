# # import json
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# #
# # # 🔧 工具函数：处理 [[[x, y, z]]] 格式
# # def unpack_point(pt):
# #     while isinstance(pt[0], list):
# #         pt = pt[0]
# #     return pt
# #
# # # 存储数据
# # data_by_epoch = {}
# #
# # # 读取 JSON 文件
# # with open("explore_goals.json", "r") as f:
# #     for line in f:
# #         entry = json.loads(line)
# #         epoch = entry["epoch"]
# #         cycle = entry["cycle"]
# #         if epoch not in data_by_epoch:
# #             data_by_epoch[epoch] = {}
# #         data_by_epoch[epoch][cycle] = {
# #             "initial": entry["initial_goals"],
# #             "desired": entry.get("desired_goals") or entry.get("\ndesired_goals"),
# #             "explore": entry.get("explore_goals") or entry.get("\nexplore_goals")
# #         }
# #
# # # 每个 epoch-cycle 单独画图
# # for epoch, cycles in data_by_epoch.items():
# #     for cycle, goals in cycles.items():
# #         fig = plt.figure()
# #         ax = fig.add_subplot(111, projection='3d')
# #
# #         inits = goals["initial"]
# #         desires = goals["desired"]
# #         explores = goals["explore"]
# #
# #         # 画 initial_goals（蓝色）
# #         for pt in inits:
# #             ax.scatter(*unpack_point(pt), color='blue', label='initial' if (epoch, cycle) == (0, 0) else "")
# #
# #         # 画 desired_goals（绿色）
# #         for pt in desires:
# #             ax.scatter(*unpack_point(pt), color='green', label='desired' if (epoch, cycle) == (0, 0) else "")
# #
# #         # 画 explore_goals（红色）
# #         for pt in explores:
# #             ax.scatter(*unpack_point(pt), color='red', label='explore' if (epoch, cycle) == (0, 0) else "")
# #
# #         ax.set_title(f"Epoch {epoch}, Cycle {cycle}")
# #         ax.set_xlabel("X")
# #         ax.set_ylabel("Y")
# #         ax.set_zlabel("Z")
# #
# #         # 避免重复图例
# #         handles, labels = ax.get_legend_handles_labels()
# #         unique = dict(zip(labels, handles))
# #         ax.legend(unique.values(), unique.keys())
# #
# #         plt.tight_layout()
# #         plt.show()
# #
# import os
# import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 🔧 工具函数：处理 [[[x, y, z]]] 格式
# def unpack_point(pt):
#     while isinstance(pt[0], list):
#         pt = pt[0]
#     return pt
#
# # 创建保存图像的文件夹
# save_dir = "figures"
# os.makedirs(save_dir, exist_ok=True)
#
# # 存储数据
# data_by_epoch = {}
#
# # 读取 JSON 文件
# with open("explore_goals.json", "r") as f:
#     for line in f:
#         entry = json.loads(line)
#         epoch = entry["epoch"]
#         cycle = entry["cycle"]
#         if epoch not in data_by_epoch:
#             data_by_epoch[epoch] = {}
#         data_by_epoch[epoch][cycle] = {
#             "initial": entry["initial_goals"],
#             "desired": entry.get("desired_goals") or entry.get("\ndesired_goals"),
#             "explore": entry.get("explore_goals") or entry.get("\nexplore_goals")
#         }
#
# # 每个 epoch-cycle 单独画图并保存
# for epoch, cycles in data_by_epoch.items():
#     for cycle, goals in cycles.items():
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#
#         inits = goals["initial"]
#         desires = goals["desired"]
#         explores = goals["explore"]
#
#         # 画 initial_goals（蓝色）
#         for pt in inits:
#             ax.scatter(*unpack_point(pt), color='blue', label='initial' if (epoch, cycle) == (0, 0) else "")
#
#         # 画 desired_goals（绿色）
#         for pt in desires:
#             ax.scatter(*unpack_point(pt), color='green', label='desired' if (epoch, cycle) == (0, 0) else "")
#
#         # 画 explore_goals（红色）
#         for pt in explores:
#             ax.scatter(*unpack_point(pt), color='red', label='explore' if (epoch, cycle) == (0, 0) else "")
#
#         ax.set_title(f"Epoch {epoch}, Cycle {cycle}")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#
#         # 避免重复图例
#         handles, labels = ax.get_legend_handles_labels()
#         unique = dict(zip(labels, handles))
#         ax.legend(unique.values(), unique.keys())
#
#         plt.tight_layout()
#
#         # 保存图像
#         filename = f"epoch_{epoch}_cycle_{cycle}.png"
#         filepath = os.path.join(save_dir, filename)
#         plt.savefig(filepath)
#         plt.close()  # 关闭当前图形窗口，节省内存
#
#
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 工具函数：处理 [[[x, y, z]]] 格式
def unpack_point(pt):
    while isinstance(pt[0], list):
        pt = pt[0]
    return pt

# 创建保存图像的文件夹
save_dir = "figures24"
os.makedirs(save_dir, exist_ok=True)

# 存储数据
data_by_epoch = {}

# 为统一坐标轴收集所有坐标点
all_points = []

# 读取 JSON 文件
with open("explore_goals17.json", "r") as f:
    for line in f:
        entry = json.loads(line)
        epoch = entry["epoch"]
        cycle = entry["cycle"]
        if epoch not in data_by_epoch:
            data_by_epoch[epoch] = {}

        initial = entry["initial_goals"]
        desired = entry.get("desired_goals") or entry.get("\ndesired_goals")
        explore = entry.get("explore_goals") or entry.get("\nexplore_goals")
        trajectories = entry.get("trajectories", [])  # optional

        #收集坐标点用于统一 axis limits
        for lst in [initial, desired]:

            all_points.extend([unpack_point(pt) for pt in lst])
        for traj in trajectories:
            all_points.extend([unpack_point(pt) for pt in traj])

        data_by_epoch[epoch][cycle] = {
            "initial": initial,
            "desired": desired,
            "explore": explore,
            "trajectories": trajectories
        }

# 计算坐标轴统一范围
all_x, all_y, all_z = zip(*all_points)
x_range = (min(all_x), max(all_x))
y_range = (min(all_y), max(all_y))
z_range = (min(all_z), max(all_z))

# 每个 epoch-cycle 单独画图并保存
for epoch, cycles in data_by_epoch.items():
    for cycle, goals in cycles.items():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        inits = goals["initial"]
        desires = goals["desired"]
        explores = goals["explore"]
        trajectories = goals.get("trajectories", [])

        for pt in inits:
            ax.scatter(*unpack_point(pt), color='blue', label='initial' if (epoch, cycle) == (0, 0) else "")
        for pt in desires:
            ax.scatter(*unpack_point(pt), color='green', label='desired' if (epoch, cycle) == (0, 0) else "")
        for pt in explores:
            ax.scatter(*unpack_point(pt), color='red', label='explore' if (epoch, cycle) == (0, 0) else "")

        # 画黑色轨迹线（黑点+线段）
        for traj in trajectories:
            traj_pts = [unpack_point(p) for p in traj]
            xs, ys, zs = zip(*traj_pts)
            ax.plot(xs, ys, zs, color='black', linewidth=1)
            ax.scatter(xs, ys, zs, color='black', s=5)

        ax.set_title(f"Epoch {epoch}, Cycle {cycle}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 坐标轴范围统一
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)

        # 图例处理
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        plt.tight_layout()
        filename = f"epoch_{epoch}_cycle_{cycle}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()


