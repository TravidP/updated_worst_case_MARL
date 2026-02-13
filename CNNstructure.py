from graphviz import Digraph

def create_cnn_actor_critic_diagram():
    # 初始化有向图
    dot = Digraph('CNN_Actor_Critic', comment='CNN Actor-Critic Network Architecture')
    
    # 全局图形设置：从上到下 (TB)，直线样式 (splines)
    dot.attr(rankdir='TB', splines='ortho', compound='true')
    dot.attr('node', fontname='Helvetica', shape='box', style='filled')
    
    # --- 颜色定义 (对应 Mermaid 中的 classDef) ---
    colors = {
        'input': {'fillcolor': '#ffffff', 'color': '#333333', 'penwidth': '2'},
        'layer': {'fillcolor': '#e1f5fe', 'color': '#01579b', 'penwidth': '2'},
        'op':    {'fillcolor': '#f5f5f5', 'color': '#9e9e9e', 'style': 'filled,dashed'},
        'actor': {'fillcolor': '#e8f5e9', 'color': '#2e7d32', 'penwidth': '2'},
        'critic':{'fillcolor': '#fff3e0', 'color': '#ef6c00', 'penwidth': '2'}
    }

    # --- 节点定义 ---
    
    # 1. Input Node (圆角矩形)
    dot.node('Input', 'Input: ob_fw\nShape: None, 200', 
             shape='note', **colors['input'])

    # 2. Reshape Operation
    dot.node('Reshape', 'Reshape to 5x5x8', 
             **colors['op'])

    # 3. Conv Layers (使用 HTML 标签实现粗体标题)
    dot.node('Conv1', '<<b>Conv2D: 32 filters</b><br/>3x3, Padding: Same, ReLU>', 
             **colors['layer'])
    
    dot.node('Conv2', '<<b>Conv2D: 64 filters</b><br/>3x3, Padding: Valid, ReLU>', 
             **colors['layer'])

    # 4. Flatten Operation
    dot.node('Flatten', 'Flatten to 576', 
             **colors['op'])

    # 5. Shared Dense Layer
    dot.node('SharedDense', '<<b>Dense Layer: 128</b><br/>Activation: ReLU>', 
             **colors['layer'])

    # 6. Branch Point (看不见的点，或者是小菱形，用于分叉)
    dot.node('Branch', 'Split', shape='diamond', height='0.5', width='0.5', 
             style='filled', fillcolor='#eeeeee', color='#999999')

    # --- Heads (Actor & Critic) ---
    
    # Policy Head
    dot.node('PiHead', '<<b>Policy Head: Dense n_a</b><br/>Softmax>', 
             **colors['actor'])
    
    # Value Head
    dot.node('VHead', '<<b>Value Head: Dense 1</b><br/>Linear>', 
             **colors['critic'])
    
    # Squeeze Op
    dot.node('Squeeze', 'Squeeze Operation', 
             **colors['op'])

    # --- Outputs ---
    # 使用平行四边形代表输出
    dot.node('PiOut', 'Output: pi', shape='parallelogram', **colors['actor'])
    dot.node('VOut', 'Output: v_preds', shape='parallelogram', **colors['critic'])

    # --- 连接逻辑 (Edges) ---
    
    dot.edge('Input', 'Reshape')
    dot.edge('Reshape', 'Conv1')
    
    # 带标签的连接
    dot.edge('Conv1', 'Conv2', label=' Shape: 5, 5, 32 ')
    dot.edge('Conv2', 'Flatten', label=' Shape: 3, 3, 64 ')
    
    dot.edge('Flatten', 'SharedDense')
    dot.edge('SharedDense', 'Branch')
    
    # 分支连接
    dot.edge('Branch', 'PiHead')
    dot.edge('Branch', 'VHead')
    
    dot.edge('PiHead', 'PiOut')
    dot.edge('VHead', 'Squeeze')
    dot.edge('Squeeze', 'VOut')

    # --- 渲染并保存 ---
    # format='png' 或 'pdf'
# 修改最后一行
    output_path = dot.render('cnn_actor_critic_flowchart', view=False, format='png')
    print(f"流程图已生成并保存至: {output_path}")

if __name__ == '__main__':
    create_cnn_actor_critic_diagram()