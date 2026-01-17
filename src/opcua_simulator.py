import asyncio
import random
import logging
from asyncua import Server

logging.basicConfig(level=logging.INFO)

async def start_simulated_plc():
    server = Server()
    await server.init()
    # FIXED: Must include port in endpoint
    server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")
    
    uri = "http://manufacturing.amrs.simulator"
    idx = await server.register_namespace(uri)

    # Create 3 Simulated Machines to fill up the charts
    machines = []
    for i in range(1, 4):
        obj = await server.nodes.objects.add_object(idx, f"Machine_{i}")
        vars = {
            'units': await obj.add_variable(idx, "UnitsProduced", 0.0),
            'defects': await obj.add_variable(idx, "DefectiveUnits", 0.0),
            'downtime': await obj.add_variable(idx, "DowntimeMinutes", 0.0)
        }
        for v in vars.values(): 
            await v.set_writable()
        machines.append(vars)

    await server.start()
    logging.info("Multi-Machine PLC Simulator started on opc.tcp://0.0.0.0:4840/freeopcua/server/")

    try:
        while True:
            for m in machines:
                await m['units'].write_value(float(random.randint(100, 500)))
                await m['defects'].write_value(float(random.randint(0, 15)))
                await m['downtime'].write_value(float(random.randint(0, 45)))
            await asyncio.sleep(2) 
    finally:
        await server.stop()
        logging.info("PLC Simulator stopped")

if __name__ == "__main__":
    asyncio.run(start_simulated_plc())