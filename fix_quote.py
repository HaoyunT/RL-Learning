# -*- coding: utf-8 -*-

# 读取文件
with open('advanced.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到引言行并加粗
for i, line in enumerate(lines):
    if '给了我们理论上的安全感' in line and '<p>"' in line and not '<strong>' in line:
        # 找到引号的位置并加粗
        lines[i] = line.replace('<p>"', '<p><strong>"').replace('"</p>', '"</strong></p>')
        print(f"✓ 第{i+1}行: 引言已加粗")
        break

# 写回文件
with open('advanced.html', 'w', encoding='utf-8', newline='') as f:
    f.writelines(lines)

print("✅ 修正完成！")
