#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : product_manager.py
@Modified By: mashenquan, 2023/11/27. Add `PrepareDocuments` action according to Section 2.2.3.5.1 of RFC 135.
"""


from metagpt.actions import UserRequirement, WritePRD
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.roles.role import Role, RoleReactMode
from metagpt.utils.common import any_to_name


class ProductManager(Role):
    """
    代表负责产品开发和管理的产品经理角色。

    属性:
        name (str): 产品经理的名字
        profile (str): 角色描述,默认为'产品经理'
        goal (str): 产品经理的目标
        constraints (str): 产品经理的约束条件和限制
    """

    name: str = "爱丽丝"
    profile: str = "产品经理"
    goal: str = "高效创建满足市场需求和用户期望的成功产品"
    constraints: str = "使用与用户需求相同的语言以实现无缝沟通"
    todo_action: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.set_actions([PrepareDocuments, WritePRD])
        self._watch([UserRequirement, PrepareDocuments])
        self.rc.react_mode = RoleReactMode.BY_ORDER
        self.todo_action = any_to_name(WritePRD)

    async def _observe(self, ignore_memory=False) -> int:
        return await super()._observe(ignore_memory=True)