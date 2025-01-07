#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
@Time    : 2023/9/4 15:40:40
@Author  : Stitch-z
@File    : tutorial_assistant.py
"""

from datetime import datetime
from typing import Dict

from metagpt.actions.write_tutorial import WriteContent, WriteDirectory
from metagpt.const import TUTORIAL_PATH
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.utils.file import File


class TutorialAssistant(Role):
    """教程助手，输入一句话即可生成 Markdown 格式的教程文档。

    参数:
        name: 角色名称
        profile: 角色描述
        goal: 角色目标
        constraints: 角色的约束条件和要求
        language: 生成教程文档使用的语言
    """

    name: str = "Stitch"
    profile: str = "Tutorial Assistant"  # 教程助手
    goal: str = "Generate tutorial documents"  # 生成教程文档
    constraints: str = (
        "Strictly follow Markdown's syntax, with neat and standardized layout"  # 严格遵循 Markdown 语法，布局整洁规范
    )
    language: str = "Chinese"  # 默认使用中文

    topic: str = ""  # 教程主题
    main_title: str = ""  # 教程主标题
    total_content: str = ""  # 存储完整的教程内容

    def __init__(self, **kwargs):
        """初始化教程助手，设置默认动作为生成目录"""
        super().__init__(**kwargs)
        self.set_actions([WriteDirectory(language=self.language)])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)  # 设置为按顺序执行动作

    async def _handle_directory(self, titles: Dict) -> Message:
        """处理教程文档的目录结构

        参数:
            titles: 包含标题和目录结构的字典，
                   格式如：{"title": "xxx", "directory": [{"dir 1": ["sub dir 1", "sub dir 2"]}]}

        返回:
            包含目录信息的消息对象
        """
        self.main_title = titles.get("title")  # 获取主标题
        directory = f"{self.main_title}\n"  # 初始化目录字符串
        self.total_content += f"# {self.main_title}"  # 将主标题添加到总内容中
        actions = list(self.actions)  # 获取当前动作列表
        # 处理一级目录
        for first_dir in titles.get("directory"):
            actions.append(WriteContent(language=self.language, directory=first_dir))  # 为每个一级目录添加写内容动作
            key = list(first_dir.keys())[0]
            directory += f"- {key}\n"
            # 处理二级目录
            for second_dir in first_dir[key]:
                directory += f"  - {second_dir}\n"
        self.set_actions(actions)  # 更新动作列表
        self.rc.max_react_loop = len(self.actions)  # 设置最大执行循环次数
        return Message()

    async def _act(self) -> Message:
        """执行角色的动作

        返回:
            包含动作执行结果的消息对象
        """
        todo = self.rc.todo  # 获取待执行的动作
        if type(todo) is WriteDirectory:  # 如果是写目录动作
            msg = self.rc.memory.get(k=1)[0]  # 获取最近的一条消息
            self.topic = msg.content  # 设置教程主题
            resp = await todo.run(topic=self.topic)  # 执行写目录动作
            logger.info(resp)
            return await self._handle_directory(resp)
        # 如果是写内容动作
        resp = await todo.run(topic=self.topic)
        logger.info(resp)
        if self.total_content != "":
            self.total_content += "\n\n\n"  # 添加内容分隔符
        self.total_content += resp  # 将新内容添加到总内容中
        return Message(content=resp, role=self.profile)

    async def react(self) -> Message:
        """响应并生成最终的教程文档

        返回:
            包含生成文档路径的消息对象
        """
        msg = await super().react()  # 执行父类的响应方法
        # 生成文档保存路径（使用当前时间戳）
        root_path = TUTORIAL_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 将总内容写入 Markdown 文件
        await File.write(root_path, f"{self.main_title}.md", self.total_content.encode("utf-8"))
        msg.content = str(root_path / f"{self.main_title}.md")  # 设置消息内容为文件路径
        return msg
