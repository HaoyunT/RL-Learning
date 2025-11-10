# -*- coding: utf-8 -*-
import re

# 读取文件
with open('advanced.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到包含第21章的行
modified = False
insight_added = False

for i, line in enumerate(lines):
    # 检查是否是引言行
    if '给了我们理论上的安全感' in line and '<p>' in line and '<strong>' not in line:
        # 加粗引言
        lines[i] = line.replace('<p>"', '<p><strong>"').replace('"</p>', '"</strong></p>')
        print(f"✓ 第{i+1}行: 引言已加粗")
        modified = True
        
        # 在下一个"一、从策略梯度说起"之前插入核心洞察
        # 找到下一个h3标签
        for j in range(i+1, min(i+20, len(lines))):
            if '<h3>一、从策略梯度说起</h3>' in lines[j]:
                # 在这之前插入核心洞察部分
                insight_section = '''
      <h3>🧩 核心洞察：PPO 是"小范围 Off-Policy"算法</h3>
      <p>虽然 PPO 通常被归类为 <strong>on-policy</strong> 算法，但严格来说，它是一种 <strong>"局部 off-policy（small off-policy）"</strong> 策略优化方法。</p>
      
      <p><strong>为什么这么说？</strong></p>
      <p>在一次训练循环中：</p>
      <ol>
        <li>我们用当前策略 $\\pi_{\\text{old}}$ 与环境交互，收集一批轨迹；</li>
        <li>然后<strong>固定这些数据</strong>，在此基础上<strong>多轮更新</strong>新策略 $\\pi_{\\theta}$。</li>
      </ol>
      <p>这意味着在优化时：$\\pi_{\\theta} \\neq \\pi_{\\text{old}}$，因此当前的优化步骤严格来说已经是 <strong>off-policy 更新</strong>。</p>
      
      <p>但 PPO 通过<strong>重要性比率</strong> $r_{\\theta}=\\frac{\\pi_{\\theta}(a|s)}{\\pi_{\\text{old}}(a|s)}$ 以及<strong>剪切约束</strong> $\\text{clip}(r_{\\theta},1-\\epsilon,1+\\epsilon)$，
      强行限制新旧策略差异在一个<strong>局部信赖域（trust region）</strong>内。</p>

      <p><strong>于是：</strong></p>
      <ul>
        <li>PPO 的更新可以被理解为一次 <strong>"小范围 off-policy、全局近似 on-policy"</strong> 的优化过程；</li>
        <li>它既允许一定程度的<strong>数据复用</strong>，又避免了 off-policy 方法常见的<strong>分布偏移问题</strong>；</li>
        <li>这也是 PPO 之所以能在<strong>"稳定性与效率之间"</strong>取得极佳平衡的原因。</li>
      </ul>

'''
                lines.insert(j, insight_section)
                print(f"✓ 第{j+1}行之前: 核心洞察部分已插入")
                insight_added = True
                break
        break

if modified or insight_added:
    # 写回文件
    with open('advanced.html', 'w', encoding='utf-8', newline='') as f:
        f.writelines(lines)
    print("\n✅ 修正完成！")
else:
    print("\n⚠️ 未找到需要修改的内容")
