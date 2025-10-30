## To use GUI for task

Enable the following in init()
```
if self.is_tasklist:
	self.taskgui=GUI_tasklist()
```

- 実行は`task_test.py`を参照

```
actions, task = policy(n_obs, env)
joint_action = {"pass": actions, "task": task}
n_obs, reward, done, info = env.step(joint_action)
```

- `"pass"`は従来のdrpの入力と同じ
- `"task"`はエージェントへのタスクの割り当てを表し，サイズはエージェント数のリストとする．
	- 割り当てはタスクリスト`env.current_tasklist`の何番目のタスクかを表す（0~）
	- 割り当てを行わない場合は`-1`とする（例：エージェント数4，`task = [-1,0,1,-1]`）

- 必要な情報
	- `env.current_tasklist`：現在のすべての未実行状態のタスクのリスト  
	（例：タスク数3，`[[1,2],[5,3],[8,9]]`）
	- `env.assigned_list`：未実行のタスクがどのエージェントに割り当てられているかのリスト．割り当てられていない場合は-1  
	（例：タスク数3，タスク2が割り当てられていない．`[1,0,-1]`）
	- `env.assigned_tasks`：各エージェントに割り当てられたタスクの情報．実行中のものも含む  
	（例：エージェント数3，エージェント2はタスクを割り当てられていない．`[[1,2],[3,4],[]]`）

