#!/usr/bin/env python3
"""
🚀 UNIFIED AI ORCHESTRATION SYSTEM
Combines ROMA, Dria, and Chief AI with IZA OS ecosystem
Provides agent tool awareness and intelligent tool selection
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import aiohttp
import sqlite3
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Tool categories for organization"""
    DATA_PROCESSING = "data_processing"
    AI_MODELS = "ai_models"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"
    BUSINESS = "business"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    RESEARCH = "research"

class AgentCapability(Enum):
    """Agent capability levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class Tool:
    """Represents a tool available to agents"""
    id: str
    name: str
    description: str
    category: ToolCategory
    capabilities: List[str]
    required_permissions: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float]
    availability: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 1.0

@dataclass
class Agent:
    """Represents an AI agent with capabilities and preferences"""
    id: str
    name: str
    role: str
    capability_level: AgentCapability
    preferred_tools: List[str]
    available_tools: List[str]
    current_task: Optional[str] = None
    status: str = "idle"
    performance_score: float = 0.0
    last_active: Optional[datetime] = None

@dataclass
class Task:
    """Represents a task that needs to be completed"""
    id: str
    description: str
    priority: int
    required_capabilities: List[str]
    suggested_tools: List[str]
    estimated_duration: int  # minutes
    dependencies: List[str]
    status: str = "pending"
    assigned_agent: Optional[str] = None
    created_at: datetime = None

class UnifiedAIOrchestrationSystem:
    """Unified system combining ROMA, Dria, Chief AI with IZA OS"""
    
    def __init__(self, base_path: str = "/Users/divinejohns/memU/memu"):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "unified_ai_orchestration.db"
        
        # Core components
        self.roma_service = None
        self.dria_service = None
        self.chief_ai = None
        self.iza_os_components = {}
        
        # Tool and agent management
        self.tools_registry: Dict[str, Tool] = {}
        self.agents_registry: Dict[str, Agent] = {}
        self.tasks_queue: List[Task] = []
        
        # IZA OS integration
        self.iza_os_manifest = {}
        self.integration_status = {}
        
        # Performance tracking
        self.performance_metrics = {}
        self.usage_statistics = {}
        
        # Initialize system
        self.init_database()
        self.load_iza_os_manifest()
        self.register_core_tools()
        self.register_core_agents()
    
    def init_database(self):
        """Initialize database for orchestration data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tools registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tools_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    required_permissions TEXT NOT NULL,
                    input_schema TEXT NOT NULL,
                    output_schema TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    availability BOOLEAN DEFAULT TRUE,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Agents registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agents_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    capability_level TEXT NOT NULL,
                    preferred_tools TEXT NOT NULL,
                    available_tools TEXT NOT NULL,
                    current_task TEXT,
                    status TEXT DEFAULT 'idle',
                    performance_score REAL DEFAULT 0.0,
                    last_active TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tasks queue table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks_queue (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    required_capabilities TEXT NOT NULL,
                    suggested_tools TEXT NOT NULL,
                    estimated_duration INTEGER NOT NULL,
                    dependencies TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    assigned_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tool usage tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_usage_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    success BOOLEAN,
                    performance_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tools_category ON tools_registry(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agents_capability ON agents_registry(capability_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks_queue(priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks_queue(status)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Unified AI Orchestration database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
    
    def load_iza_os_manifest(self):
        """Load IZA OS integration manifest"""
        try:
            manifest_path = self.base_path / "iza-os-integrated" / "00-meta" / "IZA_OS_INTEGRATION_MANIFEST.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.iza_os_manifest = json.load(f)
                logger.info("✅ Loaded IZA OS integration manifest")
            else:
                logger.warning("⚠️ IZA OS manifest not found, using default configuration")
                self.iza_os_manifest = self.get_default_iza_os_config()
                
        except Exception as e:
            logger.error(f"❌ Error loading IZA OS manifest: {e}")
            self.iza_os_manifest = self.get_default_iza_os_config()
    
    def get_default_iza_os_config(self) -> Dict[str, Any]:
        """Get default IZA OS configuration"""
        return {
            "core_components": {
                "00-meta": {"path": "iza-os-integrated/00-meta", "capabilities": ["configuration_management"]},
                "10-infra": {"path": "iza-os-integrated/10-infra", "capabilities": ["infrastructure_as_code"]},
                "20-data": {"path": "iza-os-integrated/20-data", "capabilities": ["data_processing"]},
                "30-models": {"path": "iza-os-integrated/30-models", "capabilities": ["ai_orchestration"]},
                "40-mcp-agents": {"path": "iza-os-integrated/40-mcp-agents", "capabilities": ["agent_coordination"]},
                "50-apps": {"path": "iza-os-integrated/50-apps", "capabilities": ["user_interfaces"]},
                "60-observability": {"path": "iza-os-integrated/60-observability", "capabilities": ["system_monitoring"]},
                "70-commerce-finance": {"path": "iza-os-integrated/70-commerce-finance", "capabilities": ["business_automation"]},
                "80-second-brain": {"path": "iza-os-integrated/80-second-brain", "capabilities": ["knowledge_management"]},
                "99-ops": {"path": "iza-os-integrated/99-ops", "capabilities": ["operations_management"]}
            }
        }
    
    def register_core_tools(self):
        """Register core tools from ROMA, Dria, Chief AI, and IZA OS"""
        
        # ROMA Service Tools
        roma_tools = [
            Tool(
                id="roma-etl-processor",
                name="ROMA ETL Processor",
                description="AI-powered ETL pipeline processing with intelligent data transformation",
                category=ToolCategory.DATA_PROCESSING,
                capabilities=["data_extraction", "data_transformation", "data_loading", "ai_processing"],
                required_permissions=["data_access", "processing_power"],
                input_schema={"data_source": "string", "transformation_rules": "object"},
                output_schema={"processed_data": "object", "metrics": "object"},
                performance_metrics={"throughput": 1000.0, "accuracy": 0.95, "latency": 2.5}
            ),
            Tool(
                id="roma-data-analyzer",
                name="ROMA Data Analyzer",
                description="Advanced data analysis with AI insights and pattern recognition",
                category=ToolCategory.DATA_PROCESSING,
                capabilities=["data_analysis", "pattern_recognition", "insight_generation"],
                required_permissions=["data_access", "analysis_engine"],
                input_schema={"dataset": "object", "analysis_type": "string"},
                output_schema={"analysis_results": "object", "insights": "array"},
                performance_metrics={"analysis_speed": 500.0, "accuracy": 0.92, "insight_quality": 0.88}
            )
        ]
        
        # Dria Service Tools
        dria_tools = [
            Tool(
                id="dria-knowledge-processor",
                name="Dria Knowledge Processor",
                description="Intelligent knowledge processing and semantic understanding",
                category=ToolCategory.RESEARCH,
                capabilities=["knowledge_extraction", "semantic_analysis", "knowledge_graph_building"],
                required_permissions=["knowledge_access", "processing_power"],
                input_schema={"knowledge_source": "string", "processing_type": "string"},
                output_schema={"processed_knowledge": "object", "semantic_graph": "object"},
                performance_metrics={"processing_speed": 800.0, "accuracy": 0.94, "comprehensiveness": 0.91}
            ),
            Tool(
                id="dria-intelligence-engine",
                name="Dria Intelligence Engine",
                description="Advanced intelligence processing for decision support and insights",
                category=ToolCategory.AI_MODELS,
                capabilities=["intelligence_processing", "decision_support", "insight_generation"],
                required_permissions=["intelligence_access", "decision_engine"],
                input_schema={"intelligence_query": "string", "context": "object"},
                output_schema={"intelligence_result": "object", "recommendations": "array"},
                performance_metrics={"processing_speed": 600.0, "accuracy": 0.96, "relevance": 0.93}
            )
        ]
        
        # Chief AI Tools
        chief_ai_tools = [
            Tool(
                id="chief-ai-orchestrator",
                name="Chief AI Orchestrator",
                description="Master AI orchestration system for coordinating all AI agents and workflows",
                category=ToolCategory.AI_MODELS,
                capabilities=["agent_orchestration", "workflow_management", "task_coordination"],
                required_permissions=["orchestration_access", "agent_control"],
                input_schema={"orchestration_task": "string", "agents": "array"},
                output_schema={"orchestration_result": "object", "agent_assignments": "object"},
                performance_metrics={"coordination_speed": 1200.0, "efficiency": 0.97, "success_rate": 0.95}
            ),
            Tool(
                id="chief-ai-decision-maker",
                name="Chief AI Decision Maker",
                description="High-level decision making and strategic planning for autonomous operations",
                category=ToolCategory.BUSINESS,
                capabilities=["strategic_planning", "decision_making", "resource_allocation"],
                required_permissions=["strategic_access", "decision_authority"],
                input_schema={"decision_context": "object", "options": "array"},
                output_schema={"decision": "object", "rationale": "string", "action_plan": "object"},
                performance_metrics={"decision_speed": 300.0, "accuracy": 0.98, "strategic_value": 0.94}
            )
        ]
        
        # IZA OS Integration Tools
        iza_os_tools = [
            Tool(
                id="iza-os-memory-core",
                name="IZA OS Memory Core",
                description="Central memory management and data orchestration for IZA OS ecosystem",
                category=ToolCategory.DATA_PROCESSING,
                capabilities=["memory_management", "data_orchestration", "state_management"],
                required_permissions=["memory_access", "data_control"],
                input_schema={"memory_operation": "string", "data": "object"},
                output_schema={"memory_result": "object", "state": "object"},
                performance_metrics={"memory_speed": 2000.0, "reliability": 0.99, "efficiency": 0.96}
            ),
            Tool(
                id="iza-os-command-center",
                name="IZA OS Command Center",
                description="Central command and control system for IZA OS ecosystem operations",
                category=ToolCategory.INFRASTRUCTURE,
                capabilities=["system_control", "command_execution", "status_monitoring"],
                required_permissions=["system_control", "command_authority"],
                input_schema={"command": "string", "parameters": "object"},
                output_schema={"command_result": "object", "status": "string"},
                performance_metrics={"command_speed": 1500.0, "reliability": 0.98, "response_time": 1.2}
            ),
            Tool(
                id="iza-os-monitoring",
                name="IZA OS Monitoring System",
                description="Comprehensive monitoring and observability for IZA OS ecosystem",
                category=ToolCategory.MONITORING,
                capabilities=["system_monitoring", "performance_analytics", "alert_management"],
                required_permissions=["monitoring_access", "system_visibility"],
                input_schema={"monitoring_query": "string", "time_range": "object"},
                output_schema={"monitoring_data": "object", "alerts": "array", "metrics": "object"},
                performance_metrics={"monitoring_speed": 1000.0, "accuracy": 0.97, "coverage": 0.99}
            )
        ]
        
        # Register all tools
        all_tools = roma_tools + dria_tools + chief_ai_tools + iza_os_tools
        
        for tool in all_tools:
            self.tools_registry[tool.id] = tool
            self.store_tool(tool)
        
        logger.info(f"✅ Registered {len(all_tools)} core tools")
    
    def register_core_agents(self):
        """Register core agents with tool awareness"""
        
        # ROMA Agent
        roma_agent = Agent(
            id="roma-agent",
            name="ROMA Processing Agent",
            role="Data Processing Specialist",
            capability_level=AgentCapability.EXPERT,
            preferred_tools=["roma-etl-processor", "roma-data-analyzer", "iza-os-memory-core"],
            available_tools=["roma-etl-processor", "roma-data-analyzer", "iza-os-memory-core", "iza-os-monitoring"]
        )
        
        # Dria Agent
        dria_agent = Agent(
            id="dria-agent",
            name="Dria Intelligence Agent",
            role="Knowledge & Intelligence Specialist",
            capability_level=AgentCapability.EXPERT,
            preferred_tools=["dria-knowledge-processor", "dria-intelligence-engine", "iza-os-memory-core"],
            available_tools=["dria-knowledge-processor", "dria-intelligence-engine", "iza-os-memory-core", "iza-os-monitoring"]
        )
        
        # Chief AI Agent
        chief_ai_agent = Agent(
            id="chief-ai-agent",
            name="Chief AI Orchestrator",
            role="Master AI Coordinator",
            capability_level=AgentCapability.EXPERT,
            preferred_tools=["chief-ai-orchestrator", "chief-ai-decision-maker", "iza-os-command-center"],
            available_tools=["chief-ai-orchestrator", "chief-ai-decision-maker", "iza-os-command-center", "iza-os-monitoring", "roma-etl-processor", "dria-knowledge-processor"]
        )
        
        # IZA OS Integration Agent
        iza_os_agent = Agent(
            id="iza-os-agent",
            name="IZA OS Integration Agent",
            role="Ecosystem Integration Specialist",
            capability_level=AgentCapability.ADVANCED,
            preferred_tools=["iza-os-memory-core", "iza-os-command-center", "iza-os-monitoring"],
            available_tools=["iza-os-memory-core", "iza-os-command-center", "iza-os-monitoring", "roma-etl-processor", "dria-knowledge-processor"]
        )
        
        # Register all agents
        agents = [roma_agent, dria_agent, chief_ai_agent, iza_os_agent]
        
        for agent in agents:
            self.agents_registry[agent.id] = agent
            self.store_agent(agent)
        
        logger.info(f"✅ Registered {len(agents)} core agents")
    
    def get_tool_recommendations(self, task_description: str, agent_id: str) -> List[str]:
        """Get tool recommendations for a specific task and agent"""
        try:
            agent = self.agents_registry.get(agent_id)
            if not agent:
                return []
            
            # Analyze task requirements
            task_keywords = task_description.lower().split()
            
            # Score tools based on relevance
            tool_scores = {}
            
            for tool_id, tool in self.tools_registry.items():
                if not tool.availability:
                    continue
                
                score = 0.0
                
                # Check if tool is available to agent
                if tool_id in agent.available_tools:
                    score += 0.3
                
                # Check if tool is preferred by agent
                if tool_id in agent.preferred_tools:
                    score += 0.2
                
                # Check capability match
                for keyword in task_keywords:
                    if keyword in tool.description.lower():
                        score += 0.1
                    if keyword in tool.capabilities:
                        score += 0.15
                
                # Consider performance metrics
                if tool.performance_metrics:
                    avg_performance = sum(tool.performance_metrics.values()) / len(tool.performance_metrics)
                    score += avg_performance * 0.1
                
                # Consider success rate
                score += tool.success_rate * 0.1
                
                tool_scores[tool_id] = score
            
            # Sort by score and return top recommendations
            sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
            return [tool_id for tool_id, score in sorted_tools[:5]]
            
        except Exception as e:
            logger.error(f"❌ Error getting tool recommendations: {e}")
            return []
    
    def assign_task_to_agent(self, task: Task) -> Optional[str]:
        """Assign a task to the most suitable agent"""
        try:
            best_agent = None
            best_score = 0.0
            
            for agent_id, agent in self.agents_registry.items():
                if agent.status != "idle":
                    continue
                
                score = 0.0
                
                # Check capability level match
                capability_scores = {
                    AgentCapability.BASIC: 0.2,
                    AgentCapability.INTERMEDIATE: 0.4,
                    AgentCapability.ADVANCED: 0.6,
                    AgentCapability.EXPERT: 0.8
                }
                score += capability_scores.get(agent.capability_level, 0.0)
                
                # Check if agent has required capabilities
                for required_capability in task.required_capabilities:
                    if required_capability in agent.role.lower():
                        score += 0.2
                
                # Check tool availability
                available_tools = sum(1 for tool_id in task.suggested_tools if tool_id in agent.available_tools)
                score += (available_tools / len(task.suggested_tools)) * 0.3 if task.suggested_tools else 0.0
                
                # Consider agent performance
                score += agent.performance_score * 0.1
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            if best_agent:
                # Update agent status
                self.agents_registry[best_agent].status = "busy"
                self.agents_registry[best_agent].current_task = task.id
                self.agents_registry[best_agent].last_active = datetime.now()
                
                # Update task
                task.assigned_agent = best_agent
                task.status = "assigned"
                
                self.store_agent(self.agents_registry[best_agent])
                self.store_task(task)
                
                logger.info(f"✅ Task {task.id} assigned to agent {best_agent}")
                return best_agent
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error assigning task: {e}")
            return None
    
    def execute_task_with_tools(self, task_id: str, agent_id: str, selected_tools: List[str]) -> Dict[str, Any]:
        """Execute a task using selected tools"""
        try:
            task = next((t for t in self.tasks_queue if t.id == task_id), None)
            agent = self.agents_registry.get(agent_id)
            
            if not task or not agent:
                return {"success": False, "error": "Task or agent not found"}
            
            execution_result = {
                "task_id": task_id,
                "agent_id": agent_id,
                "selected_tools": selected_tools,
                "start_time": datetime.now(),
                "tool_results": {},
                "success": True
            }
            
            # Execute each selected tool
            for tool_id in selected_tools:
                if tool_id not in self.tools_registry:
                    continue
                
                tool = self.tools_registry[tool_id]
                
                # Track tool usage
                self.track_tool_usage(tool_id, agent_id, task_id)
                
                # Simulate tool execution (in real implementation, this would call actual tools)
                tool_result = self.simulate_tool_execution(tool, task)
                execution_result["tool_results"][tool_id] = tool_result
                
                # Update tool usage statistics
                tool.usage_count += 1
                tool.last_used = datetime.now()
                self.store_tool(tool)
            
            # Update task status
            task.status = "completed"
            execution_result["end_time"] = datetime.now()
            execution_result["duration"] = (execution_result["end_time"] - execution_result["start_time"]).total_seconds()
            
            # Update agent status
            agent.status = "idle"
            agent.current_task = None
            agent.performance_score = min(1.0, agent.performance_score + 0.1)
            
            self.store_task(task)
            self.store_agent(agent)
            
            logger.info(f"✅ Task {task_id} completed by agent {agent_id} using tools: {selected_tools}")
            return execution_result
            
        except Exception as e:
            logger.error(f"❌ Error executing task: {e}")
            return {"success": False, "error": str(e)}
    
    def simulate_tool_execution(self, tool: Tool, task: Task) -> Dict[str, Any]:
        """Simulate tool execution (placeholder for real implementation)"""
        import random
        
        # Simulate execution time based on tool performance
        execution_time = random.uniform(0.5, 3.0)
        time.sleep(execution_time)
        
        # Simulate success based on tool success rate
        success = random.random() < tool.success_rate
        
        return {
            "tool_id": tool.id,
            "tool_name": tool.name,
            "execution_time": execution_time,
            "success": success,
            "result": f"Tool {tool.name} executed successfully" if success else f"Tool {tool.name} execution failed",
            "performance_score": random.uniform(0.7, 1.0) if success else random.uniform(0.0, 0.5)
        }
    
    def track_tool_usage(self, tool_id: str, agent_id: str, task_id: str):
        """Track tool usage for analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tool_usage_tracking 
                (tool_id, agent_id, task_id, start_time, success, performance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                tool_id,
                agent_id,
                task_id,
                datetime.now(),
                True,  # Will be updated when task completes
                1.0    # Will be updated with actual performance
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error tracking tool usage: {e}")
    
    def get_agent_tool_awareness(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive tool awareness for an agent"""
        try:
            agent = self.agents_registry.get(agent_id)
            if not agent:
                return {}
            
            awareness = {
                "agent_info": {
                    "id": agent.id,
                    "name": agent.name,
                    "role": agent.role,
                    "capability_level": agent.capability_level.value,
                    "status": agent.status,
                    "performance_score": agent.performance_score
                },
                "available_tools": [],
                "preferred_tools": [],
                "tool_capabilities": {},
                "recommendations": {}
            }
            
            # Get detailed tool information
            for tool_id in agent.available_tools:
                if tool_id in self.tools_registry:
                    tool = self.tools_registry[tool_id]
                    tool_info = {
                        "id": tool.id,
                        "name": tool.name,
                        "description": tool.description,
                        "category": tool.category.value,
                        "capabilities": tool.capabilities,
                        "performance_metrics": tool.performance_metrics,
                        "success_rate": tool.success_rate,
                        "usage_count": tool.usage_count,
                        "availability": tool.availability
                    }
                    
                    awareness["available_tools"].append(tool_info)
                    
                    if tool_id in agent.preferred_tools:
                        awareness["preferred_tools"].append(tool_info)
                    
                    # Build capability mapping
                    for capability in tool.capabilities:
                        if capability not in awareness["tool_capabilities"]:
                            awareness["tool_capabilities"][capability] = []
                        awareness["tool_capabilities"][capability].append(tool_info)
            
            # Generate recommendations for common task types
            common_tasks = [
                "data processing",
                "knowledge analysis",
                "system monitoring",
                "decision making",
                "workflow orchestration"
            ]
            
            for task_type in common_tasks:
                recommendations = self.get_tool_recommendations(task_type, agent_id)
                awareness["recommendations"][task_type] = recommendations
            
            return awareness
            
        except Exception as e:
            logger.error(f"❌ Error getting agent tool awareness: {e}")
            return {}
    
    def store_tool(self, tool: Tool):
        """Store tool in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tools_registry 
                (id, name, description, category, capabilities, required_permissions, 
                 input_schema, output_schema, performance_metrics, availability, 
                 last_used, usage_count, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tool.id,
                tool.name,
                tool.description,
                tool.category.value,
                json.dumps(tool.capabilities),
                json.dumps(tool.required_permissions),
                json.dumps(tool.input_schema),
                json.dumps(tool.output_schema),
                json.dumps(tool.performance_metrics),
                tool.availability,
                tool.last_used,
                tool.usage_count,
                tool.success_rate
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing tool: {e}")
    
    def store_agent(self, agent: Agent):
        """Store agent in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agents_registry 
                (id, name, role, capability_level, preferred_tools, available_tools, 
                 current_task, status, performance_score, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent.id,
                agent.name,
                agent.role,
                agent.capability_level.value,
                json.dumps(agent.preferred_tools),
                json.dumps(agent.available_tools),
                agent.current_task,
                agent.status,
                agent.performance_score,
                agent.last_active
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing agent: {e}")
    
    def store_task(self, task: Task):
        """Store task in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tasks_queue 
                (id, description, priority, required_capabilities, suggested_tools, 
                 estimated_duration, dependencies, status, assigned_agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id,
                task.description,
                task.priority,
                json.dumps(task.required_capabilities),
                json.dumps(task.suggested_tools),
                task.estimated_duration,
                json.dumps(task.dependencies),
                task.status,
                task.assigned_agent,
                task.created_at or datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing task: {e}")
    
    def generate_orchestration_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration report"""
        try:
            report = {
                "system_status": {
                    "timestamp": datetime.now().isoformat(),
                    "total_tools": len(self.tools_registry),
                    "total_agents": len(self.agents_registry),
                    "pending_tasks": len([t for t in self.tasks_queue if t.status == "pending"]),
                    "active_tasks": len([t for t in self.tasks_queue if t.status == "assigned"]),
                    "completed_tasks": len([t for t in self.tasks_queue if t.status == "completed"])
                },
                "tools_summary": {
                    "by_category": {},
                    "performance_metrics": {},
                    "usage_statistics": {}
                },
                "agents_summary": {
                    "by_capability": {},
                    "performance_scores": {},
                    "current_status": {}
                },
                "iza_os_integration": {
                    "components": list(self.iza_os_manifest.get("core_components", {}).keys()),
                    "integration_status": "operational",
                    "communication_patterns": self.iza_os_manifest.get("communication_patterns", {})
                },
                "recommendations": {
                    "tool_optimizations": [],
                    "agent_improvements": [],
                    "system_enhancements": []
                }
            }
            
            # Tools summary by category
            for tool in self.tools_registry.values():
                category = tool.category.value
                if category not in report["tools_summary"]["by_category"]:
                    report["tools_summary"]["by_category"][category] = 0
                report["tools_summary"]["by_category"][category] += 1
            
            # Agents summary by capability
            for agent in self.agents_registry.values():
                capability = agent.capability_level.value
                if capability not in report["agents_summary"]["by_capability"]:
                    report["agents_summary"]["by_capability"][capability] = 0
                report["agents_summary"]["by_capability"][capability] += 1
                
                report["agents_summary"]["performance_scores"][agent.id] = agent.performance_score
                report["agents_summary"]["current_status"][agent.id] = agent.status
            
            # Save report
            report_path = self.base_path / "unified_ai_orchestration_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Orchestration report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating orchestration report: {e}")
            return {}

async def main():
    """Main execution function"""
    logger.info("🚀 Starting Unified AI Orchestration System...")
    
    # Initialize system
    orchestration_system = UnifiedAIOrchestrationSystem()
    
    try:
        # Demonstrate tool awareness
        logger.info("📋 Demonstrating agent tool awareness...")
        
        for agent_id in orchestration_system.agents_registry.keys():
            awareness = orchestration_system.get_agent_tool_awareness(agent_id)
            logger.info(f"Agent {agent_id} tool awareness: {len(awareness.get('available_tools', []))} tools available")
        
        # Create sample task
        sample_task = Task(
            id="sample-task-001",
            description="Process enterprise data and generate intelligence insights",
            priority=1,
            required_capabilities=["data_processing", "intelligence_analysis"],
            suggested_tools=["roma-etl-processor", "dria-intelligence-engine"],
            estimated_duration=30,
            dependencies=[],
            created_at=datetime.now()
        )
        
        orchestration_system.tasks_queue.append(sample_task)
        
        # Assign task
        assigned_agent = orchestration_system.assign_task_to_agent(sample_task)
        if assigned_agent:
            logger.info(f"✅ Task assigned to agent: {assigned_agent}")
            
            # Get tool recommendations
            recommendations = orchestration_system.get_tool_recommendations(sample_task.description, assigned_agent)
            logger.info(f"🔧 Tool recommendations: {recommendations}")
            
            # Execute task with recommended tools
            result = orchestration_system.execute_task_with_tools(sample_task.id, assigned_agent, recommendations[:2])
            logger.info(f"✅ Task execution result: {result['success']}")
        
        # Generate report
        report = orchestration_system.generate_orchestration_report()
        logger.info(f"📊 System status: {report['system_status']}")
        
        logger.info("✅ Unified AI Orchestration System demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
