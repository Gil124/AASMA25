import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  IconButton,
  TextField,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import {
  Delete as DeleteIcon,
  Add as AddIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
} from "@mui/icons-material";
import { GridLoader } from "react-spinners";
import { createGame } from "../utils/apiClient";

import "./SimulationConfig.scss";

// Available player types based on the codebase analysis
const PLAYER_TYPES = [
  {
    value: "RANDOM",
    label: "Random Player",
    description: "Chooses actions at random",
  },
  {
    value: "CATANATRON",
    label: "Catanatron (AlphaBeta)",
    description: "Advanced AI using Alpha-Beta search algorithm",
  },
  {
    value: "ALPHABETA_IMPROVED",
    label: "AlphaBeta with Improved Pruning",
    description: "Enhanced Alpha-Beta search with advanced pruning techniques",
  },
  {
    value: "WEIGHTED_RANDOM",
    label: "Weighted Random",
    description: "Random but favors building cities/settlements",
  },
  {
    value: "VALUE_FUNCTION",
    label: "Value Function",
    description: "Uses hand-crafted value function",
  },
  {
    value: "MCTS",
    label: "MCTS Player",
    description: "Monte Carlo Tree Search algorithm",
  },
  {
    value: "GREEDY_PLAYOUTS",
    label: "Greedy Playouts",
    description: "Plays random games to evaluate moves",
  },
  {
    value: "VICTORY_POINT",
    label: "Victory Point",
    description: "Focuses on immediate victory point gains",
  },
  {
    value: "RL_DQN",
    label: "RL Agent (DQN)",
    description: "Deep Q-Network reinforcement learning agent",
  },
  {
    value: "RL_PPO",
    label: "RL Agent (PPO)",
    description: "Proximal Policy Optimization reinforcement learning agent",
  },
  {
    value: "RL_VALUE",
    label: "RL Agent (Value Function)",
    description: "Value function-based reinforcement learning agent",
  },
  {
    value: "RL_TENSOR",
    label: "RL Agent (TensorFlow)",
    description: "TensorFlow-based reinforcement learning agent",
  },
];

const PLAYER_COLORS = ["RED", "BLUE", "ORANGE", "WHITE"];

export default function SimulationConfig() {
  const navigate = useNavigate();
  const [players, setPlayers] = useState([
    { type: "RANDOM", color: "RED", params: {} },
    { type: "CATANATRON", color: "BLUE", params: {} },
  ]);
  const [loading, setLoading] = useState(false);
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);

  const addPlayer = () => {
    if (players.length < 4) {
      const availableColors = PLAYER_COLORS.filter(
        (color) => !players.some((player) => player.color === color)
      );
      setPlayers([
        ...players,
        {
          type: "RANDOM",
          color: availableColors[0] || PLAYER_COLORS[players.length],
          params: {},
        },
      ]);
    }
  };

  const removePlayer = (index) => {
    if (players.length > 2) {
      setPlayers(players.filter((_, i) => i !== index));
    }
  };

  const updatePlayerType = (index, newType) => {
    const newPlayers = [...players];
    newPlayers[index].type = newType;
    // Reset params when changing type
    newPlayers[index].params = getDefaultParams(newType);
    setPlayers(newPlayers);
  };

  const updatePlayerColor = (index, newColor) => {
    const newPlayers = [...players];
    newPlayers[index].color = newColor;
    setPlayers(newPlayers);
  };

  const updatePlayerParam = (index, paramKey, paramValue) => {
    const newPlayers = [...players];
    newPlayers[index].params = {
      ...newPlayers[index].params,
      [paramKey]: paramValue,
    };
    setPlayers(newPlayers);
  };

  const getDefaultParams = (playerType) => {
    switch (playerType) {
      case "ALPHABETA_IMPROVED":
        return { depth: 2, epsilon: 0.1 };
      case "RL_DQN":
        return {
          modelPath: "",
          epsilon: 0.1,
          learningRate: 0.0003,
          trainingMode: false,
        };
      case "RL_PPO":
        return {
          modelPath: "",
          learningRate: 0.0003,
          clipRatio: 0.2,
          trainingMode: false,
        };
      case "RL_VALUE":
        return { modelPath: "", epsilon: 0.2 };
      case "RL_TENSOR":
        return { modelPath: "" };
      case "MCTS":
        return { numSimulations: 100, prunning: false };
      case "GREEDY_PLAYOUTS":
        return { numPlayouts: 25 };
      default:
        return {};
    }
  };

  const getPlayerTypeInfo = (type) => {
    return PLAYER_TYPES.find((pt) => pt.value === type) || PLAYER_TYPES[0];
  };

  const handleStartSimulation = async () => {
    setLoading(true);
    try {
      // Convert player configurations to the format expected by the API
      // For now, map them to supported types in the backend
      const playerKeys = players.map((player) => {
        switch (player.type) {
          case "RANDOM":
          case "WEIGHTED_RANDOM":
            return "RANDOM";
          case "ALPHABETA_IMPROVED":
          case "VALUE_FUNCTION":
          case "MCTS":
          case "GREEDY_PLAYOUTS":
          case "VICTORY_POINT":
          case "CATANATRON":
          case "RL_DQN":
          case "RL_PPO":
          case "RL_VALUE":
          case "RL_TENSOR":
          default:
            return "CATANATRON";
        }
      });

      const gameId = await createGame(playerKeys);
      navigate("/games/" + gameId);
    } catch (error) {
      console.error("Failed to create simulation:", error);
      setLoading(false);
    }
  };

  const getAvailableColors = (currentIndex) => {
    return PLAYER_COLORS.filter(
      (color) =>
        !players.some(
          (player, index) => index !== currentIndex && player.color === color
        )
    );
  };

  const renderPlayerParams = (player, playerIndex) => {
    const params = player.params || {};
    const paramKeys = Object.keys(params);

    if (paramKeys.length === 0) {
      return null;
    }

    return (
      <Box className="player-params">
        {paramKeys.map((paramKey) => {
          const value = params[paramKey];

          if (typeof value === "boolean") {
            return (
              <FormControlLabel
                key={paramKey}
                control={
                  <Switch
                    checked={value}
                    onChange={(e) =>
                      updatePlayerParam(playerIndex, paramKey, e.target.checked)
                    }
                    size="small"
                  />
                }
                label={formatParamLabel(paramKey)}
                className="param-switch"
              />
            );
          } else if (typeof value === "number") {
            return (
              <TextField
                key={paramKey}
                label={formatParamLabel(paramKey)}
                type="number"
                value={value}
                onChange={(e) =>
                  updatePlayerParam(
                    playerIndex,
                    paramKey,
                    parseFloat(e.target.value) || 0
                  )
                }
                size="small"
                className="param-input"
                inputProps={{
                  step:
                    paramKey.includes("Rate") ||
                    paramKey.includes("epsilon") ||
                    paramKey.includes("Ratio")
                      ? 0.001
                      : 1,
                  min: 0,
                }}
              />
            );
          } else {
            return (
              <TextField
                key={paramKey}
                label={formatParamLabel(paramKey)}
                value={value}
                onChange={(e) =>
                  updatePlayerParam(playerIndex, paramKey, e.target.value)
                }
                size="small"
                className="param-input"
                placeholder={
                  paramKey === "modelPath" ? "Enter model file path..." : ""
                }
              />
            );
          }
        })}
      </Box>
    );
  };

  const formatParamLabel = (paramKey) => {
    return paramKey
      .replace(/([A-Z])/g, " $1")
      .replace(/^./, (str) => str.toUpperCase())
      .replace(/Path$/, " Path")
      .replace(/Mode$/, " Mode");
  };

  return (
    <div className="simulation-config">
      <div className="config-header">
        <Button
          variant="outlined"
          onClick={() => navigate("/")}
          className="back-button"
        >
          ← Back to Home
        </Button>
        <Typography variant="h3" component="h1" className="title">
          Configure Simulation
        </Typography>
        <Typography variant="subtitle1" className="subtitle">
          Set up a game with AI agents to watch them compete
        </Typography>
      </div>

      <div className="config-content">
        {!loading ? (
          <>
            <div className="players-section">
              <div className="section-header">
                <Typography variant="h5">
                  Players ({players.length}/4)
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={addPlayer}
                  disabled={players.length >= 4}
                  startIcon={<AddIcon />}
                  size="small"
                >
                  Add Player
                </Button>
              </div>

              <div className="players-list">
                {players.map((player, index) => (
                  <Card key={index} className="player-card">
                    <CardContent>
                      <div className="player-card-header">
                        <Chip
                          label={player.color}
                          className={`color-chip color-${player.color.toLowerCase()}`}
                          size="small"
                        />
                        <Typography variant="h6">Player {index + 1}</Typography>
                        {players.length > 2 && (
                          <IconButton
                            size="small"
                            onClick={() => removePlayer(index)}
                            className="remove-button"
                          >
                            <DeleteIcon />
                          </IconButton>
                        )}
                      </div>

                      <div className="player-controls">
                        <FormControl fullWidth margin="normal">
                          <InputLabel>Player Type</InputLabel>
                          <Select
                            value={player.type}
                            onChange={(e) =>
                              updatePlayerType(index, e.target.value)
                            }
                            label="Player Type"
                          >
                            {PLAYER_TYPES.map((type) => (
                              <MenuItem key={type.value} value={type.value}>
                                {type.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>

                        <FormControl fullWidth margin="normal">
                          <InputLabel>Color</InputLabel>
                          <Select
                            value={player.color}
                            onChange={(e) =>
                              updatePlayerColor(index, e.target.value)
                            }
                            label="Color"
                          >
                            {getAvailableColors(index).map((color) => (
                              <MenuItem key={color} value={color}>
                                <div className="color-option">
                                  <div
                                    className={`color-dot color-${color.toLowerCase()}`}
                                  ></div>
                                  {color}
                                </div>
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </div>

                      <Typography
                        variant="body2"
                        className="player-description"
                      >
                        {getPlayerTypeInfo(player.type).description}
                      </Typography>

                      {/* Parameter Configuration */}
                      {Object.keys(player.params).length > 0 && (
                        <Accordion className="player-params-accordion">
                          <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            className="params-header"
                          >
                            <SettingsIcon sx={{ mr: 1, fontSize: 16 }} />
                            <Typography variant="body2">
                              Advanced Configuration
                            </Typography>
                          </AccordionSummary>
                          <AccordionDetails className="params-content">
                            {renderPlayerParams(player, index)}
                          </AccordionDetails>
                        </Accordion>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            <div className="simulation-controls">
              <Typography variant="h6" className="controls-title">
                Simulation Settings
              </Typography>
              <Typography variant="body2" className="controls-description">
                Open hand gameplay with automatic bot decisions
              </Typography>

              <Button
                variant="contained"
                color="primary"
                size="large"
                onClick={handleStartSimulation}
                className="start-button"
                disabled={players.length < 2}
              >
                Start Simulation
              </Button>
            </div>
          </>
        ) : (
          <div className="loading-container">
            <GridLoader color="#1976d2" size={15} />
            <Typography
              variant="h6"
              style={{ marginTop: "20px", color: "white" }}
            >
              Starting simulation...
            </Typography>
          </div>
        )}
      </div>
    </div>
  );
}
