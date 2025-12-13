将你本地的 RKNN 安装包复制到本目录（assets/）以便通过网站提供下载。

需要放置的精确文件名：
- rknn_toolkit_lite2-1.5.2-cp310-cp310-linux_aarch64.whl  （板子端 runtime）
- rknn_toolkit2-1.5.2+b642f30c-cp310-cp310-linux_x86_64.whl  （PC 端编译工具）

使用方法：
1. 将上述两个文件复制到本仓库的 assets/ 文件夹。
2. 在项目根目录打开终端，运行：

   python -m http.server 8000

3. 在浏览器打开：

   http://localhost:8000/deploy.html

此页面将显示下载按钮，点击即可下载对应文件到浏览器默认下载目录。