# 图片占位符说明

本文需要补充以下截图：

## 1. triton-compilation-flow.png
**描述**: Triton编译流程图
**内容**: 展示 Python → TTIR → TTGIR → LLIR → PTX → SASS 的完整流程
**建议**: 可以用Mermaid或Draw.io绘制

## 2. gpu-optimization.png
**描述**: GPU编程优化示意图
**内容**: 展示GPU内存层次、warp组织等基本概念

## 3. modal-run-result.png
**描述**: Modal运行Vector Add的输出
**内容**: 运行成功的日志输出

## 4. ir-viewer-demo.png
**描述**: IR Viewer工具演示
**内容**: 展示TTIR和Python代码的绑定关系，左右面板交互

## 5. online-tool.png
**描述**: 在线工具页面
**内容**: deciding.github.io/txl/tools/ir-viewer.html 页面截图

---

## 截图位置参考

在文章中搜索 `![...- 待补充]` 可以找到所有需要替换的位置：

```bash
grep -n "待补充" zhihu/triton-series-01-vector-add.md
```

---

## 建议的截图尺寸

- 宽度: 800-1200px
- 格式: PNG或JPG
- 命名: 使用英文下划线分隔
