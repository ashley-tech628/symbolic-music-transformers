# 求职展示建议

项目作者：Ashley Liu (Xinying Liu)，独立完成。

建议仓库名：`symbolic-music-transformers`。主要适合 MLE / AI Engineer；SWE 展示重点是模块化代码、命令行入口、实验产物和自动化检查。

可用于英文简历的描述：

> Built decoder-only and encoder-decoder Transformers for four-part symbolic music generation and melody-conditioned harmonization, with transposition augmentation, baseline evaluation, and MIDI generation.

如需补充指标：

> Recorded 31.8% teacher-forced harmony-token accuracy on a monitored held-out split in a saved 30-epoch experiment.

不要写“音乐质量提升六倍”或“测试集全面超过基线”：历史 baseline 的评估处理不完全一致，预测准确率也不等于试听质量。

README、技术报告、原始实验输出、清理后的 notebook、MIDI/WAV 试听和测试已整理。打包辅助新增的功能与原始独立项目贡献分别说明在 ATTRIBUTION.md。当前还未推送到 GitHub；完整训练管线尚需在安装依赖的环境中验证。
