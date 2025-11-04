import cv2
import numpy as np


def restore_masks_to_image(mask_data, image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 将掩码数据还原到图像上
    for mask in mask_data:
        values = list(map(float, mask.split()))
        class_id = int(values[0])
        mask_values = values[1:]

        # 将掩码数据转换为NumPy数组
        mask_array = np.array(mask_values, dtype=np.float32).reshape((int(len(mask_values) / 2), 2))

        # 将相对于图像大小的百分比转换为具体坐标值
        mask_array[:, 0] *= img.shape[1]  # 宽度
        mask_array[:, 1] *= img.shape[0]  # 高度

        # 将坐标值转换为整数
        mask_array = mask_array.astype(np.int32)

        # 在图像上绘制掩码
        cv2.polylines(img, [mask_array], isClosed=True, color=(0, 255, 0), thickness=2)

        # 在图像上绘制每个坐标点
        for point in mask_array:
            cv2.circle(img, tuple(point), 3, (255, 0, 0), -1)  # -1 表示填充圆

    # 保存带有掩码和坐标点的图像
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    with open(r'E:\pycharm\DataTransformer\data\label2\aug_geo_made_menshouba2_8_3_12.txt', 'r') as file:
        mask_data = [line.strip() for line in file.readlines()]
    #mask_data = [
   #     "22 0.00746875 0.0539294 0.117891 0.0921412 0.231297 0.110118 0.2895 0.0674118 0.331281 0.0472 0.3865 0.0696706 0.423813 0.0943765 0.446188 0.105624 0.467078 0.1528 0.517813 0.182024 0.577516 0.253929 0.658094 0.379765 0.690922 0.532588 0.687937 0.6 0.650625 0.555059 0.658094 0.644941 0.668547 0.755059 0.676 0.838212 0.658094 0.894376 0.613328 0.925835 0.589453 0.914612 0.590938 0.856188 0.552141 0.791012 0.523781 0.725835 0.528266 0.633718 0.498422 0.577529 0.444703 0.505624 0.407391 0.505624 0.395453 0.541576 0.417844 0.591012 0.450672 0.642706 0.456641 0.642706 0.461109 0.725835 0.458125 0.786518 0.450672 0.853929 0.444703 0.898871 0.401422 0.869671 0.411875 0.815741 0.423813 0.734824 0.425297 0.694376 0.361125 0.608988 0.316359 0.588753 0.280547 0.703365 0.271594 0.757294 0.261141 0.829224 0.268609 0.869671 0.277562 0.901129 0.250703 0.937082 0.222344 0.939318 0.231297 0.901129 0.222344 0.844941 0.238766 0.7236 0.246219 0.642706 0.271594 0.510118 0.182062 0.507859 0.0999844 0.525835 0.0208906 0.494376 0.0015 0.0516941"
   # ]

    image_path = r"E:\pycharm\DataTransformer\data\image2\aug_geo_made_menshouba2_8_3_12.jpg"
    output_path = "1_out.jpg"

    restore_masks_to_image(mask_data, image_path, output_path)
