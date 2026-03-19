# Astar Island - Simulation Mechanics

## Map Generation

Maps are generated deterministically from **seeds**. Each round uses 5 different seeds.

### Terrain Features

- **Ocean**: Borders the island and forms fjords
- **Fjords**: Water channels cutting into the landmass
- **Mountain chains**: High terrain barriers
- **Forest patches**: Wooded areas that grow and reclaim land
- **Settlements**: Viking villages that grow, trade, and fight

## Annual Simulation Phases

The simulation runs for 50 years. Each year consists of 5 phases executed in order:

### 1. Growth Phase

- Settlements produce **food** based on surrounding terrain
- Settlements **expand** when population and food thresholds are met
- Coastal settlements build **ports**
- Settlements with ports build **longships**

### 2. Conflict Phase

- Settlements may **raid** neighboring settlements
- **Longship range**: Settlements with longships can raid across water
- Raids transfer wealth and can destroy weaker settlements

### 3. Trade Phase

- Settlements with **ports** exchange goods
- Trade increases wealth and food for participating settlements

### 4. Winter Phase

- Settlements lose **food** proportional to population
- Settlements that cannot sustain their population **collapse to Ruins**

### 5. Environment Phase

- **Ruins** are gradually **reclaimed** by forest or converted to outposts
- Forest spreads into adjacent empty terrain

## Settlement Properties

Each settlement tracks:

| Property | Description |
|----------|-------------|
| position | Grid coordinates (x, y) |
| population | Number of inhabitants |
| food | Food reserves |
| wealth | Accumulated resources |
| defense | Military strength |
| tech | Technology level |
| port | Whether settlement has a port |
| longship | Whether settlement has a longship |
| faction | Which faction the settlement belongs to |
