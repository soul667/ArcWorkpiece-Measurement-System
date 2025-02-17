@/antd-demo/src/component/FilterSection.js 中添加优化以下功能

1. 添加快捷键 X Y Z 分别文其为切换 x_mode,y_mode,z_mode ...
2. 将x_mode,y_mode,z_mode的下拉选项都减小体积，现在 了
3. 在点击裁减后，将所有区域信息清零，并且重新加载图像
3. 添加去噪按钮，红色按钮并且添加适当图标，去噪按钮左边添加 (nb_neighbors=100, std_ratio=0.5)的输入框（用其中文），请求到相应的后端接口
4. 添加显示点云复选框  这个参数在裁减的时候一块上传

@/UserInterface/fastapi_test.py 中添加denose的接口，使用remove_statistical_outlier