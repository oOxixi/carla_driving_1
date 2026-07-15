# A/C 运行与验证

不启动 CARLA 时，运行完整的 CARLA-free 回归：

```powershell
conda run --no-capture-output -n carla python -m pytest car_control_A/tests car_control_C/tests -q
```

先在另一个 PowerShell 启动已安装的 CARLA 0.9.16（不由测试自动启动）：

```powershell
cd F:\carla_driving_rstar\CARLA_0.9.16
.\CarlaUE4.exe -quality-level=Low -carla-port=2000
```

服务稳定后，在项目根目录运行可选的会话/Actor 清理烟测：

```powershell
$env:CARLA_SMOKE = "1"
conda run --no-capture-output -n carla python -m pytest car_control_A/tests/test_simulator_smoke.py -q
```

烟测只连接 `127.0.0.1:2000`，在当前地图临时生成一个 Ego，退出时销毁该 Actor 并恢复 World 设置及 Traffic Manager 的异步模式；不加载地图、不启动服务，也不依赖 B/D。
