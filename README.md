# Elevemos el nivel, hablemos de Patrones Agénticos con ADK de Google (Python)

La Inteligencia Artificial ha evolucionado. Ya no estamos limitados a enviar un *prompt* a un chatbot y cruzar los dedos esperando una buena respuesta. Hemos entrado en la era de la **IA Agéntica**, donde construimos "equipos" de agentes de IA autónomos que colaboran, se corrigen a sí mismos y ejecutan flujos de trabajo complejos.

Pero, ¿cómo orquestamos este caos de múltiples LLMs (Grandes Modelos de Lenguaje) sin volvernos locos con el código espagueti?

Aquí es donde entra **Agent Development Kit (ADK)** de Google. El ADK es un framework de código abierto diseñado para simplificar y estandarizar la creación de agentes. Proporciona una arquitectura modular para manejar LLMs, memoria y herramientas, permitiéndonos enfocarnos en la lógica de negocio a través de **Patrones de Orquestación**.

En este tutorial, exploraremos los 4 patrones agénticos más importantes y cómo implementarlos en **Python** usando Gemini 3.5 Flash Lite y `google-adk` (`>=2.15.1`).

---

## Inicio Rápido

1. **Crear y activar el entorno virtual (Python >= 3.11):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Configurar tu API Key:**

```bash
cp .env.example .env
# Edita .env y agrega tu GOOGLE_API_KEY
```

3. **Ejecutar los agentes con ADK:**

```bash
# Interfaz web interactiva (detecta los 4 paquetes automáticamente)
adk web

# O ejecutar un patrón específico desde la terminal
adk run p01_router
adk run p02_sequential
adk run p03_loop
adk run p04_parallel
```

---

## 1. El Patrón Enrutador (Router Agent): El Controlador de Tráfico Inteligente

Un agente orquestador que analiza la intención del usuario y delega la tarea al sub-agente más especializado, actuando como un "controlador de tráfico" inteligente.

### **¿Por qué es útil este patrón?**

* Delegación Inteligente: Usa la `description` de cada sub-agente para decidir a quién enviar la solicitud.
* Modular y Especializado: Permite construir sistemas complejos a partir de agentes simples y enfocados en una sola tarea.
* Eficiencia de Costos: Habilita el uso de LLMs más pequeños y económicos para tareas especializadas y con reglas específicas.

Imagina que un cliente escribe a tu empresa: *"Me cobraron de más el mes pasado y además la app de mi teléfono se cierra sola"*. Un operador humano tendría que leer esto y transferir el ticket a dos departamentos distintos. El **Agente Enrutador** automatiza exactamente esto.

Actúa como un orquestador que analiza la intención del usuario y delega silenciosamente la tarea al sub-agente más especializado, basándose únicamente en la propiedad `description` de cada uno.

### Casos de Uso Empresariales
* **Atención al Cliente "Zero-Touch":** Triage inteligente de tickets (Soporte Técnico vs. Facturación).
* **Conserje de E-commerce:** Unifica intenciones de compra (Agente de Catálogo) con consultas de postventa (Agente de Logística) en un solo chat.

### El Código (`p01_router/agent.py`)

En este ejemplo de un Bot de Seguros, el Router decide a qué especialista enviar la consulta del cliente:

```python
from google.adk.agents import LlmAgent

GEMINI_MODEL = "gemini-3.5-flash-lite"

# 1. Agentes Especializados
car_insurance_agent = LlmAgent(
    name="CarInsuranceAgent",
    model=GEMINI_MODEL,
    # La descripción es CRÍTICA. El Router la lee para tomar decisiones.
    description="Expert exclusively in Car insurance, buying new or used cars, car accidents, roadside assistance, and deductibles.",
    instruction="You are a Car Insurance Specialist... Output *only* your response as an expert agent.",
)

life_insurance_agent = LlmAgent(
    name="LifeInsuranceAgent",
    model=GEMINI_MODEL,
    description="Expert in life insurance, death policies, savings for the future, and family protection.",
    instruction="You are a Life Insurance Specialist... Output *only* your response as an expert agent.",
)

health_insurance_agent = LlmAgent(
    name="HealthInsuranceAgent",
    model=GEMINI_MODEL,
    description="Expert in supplementary health insurance, medical expense reimbursements, clinics, and pharmacies.",
    instruction="You are a Supplementary Health Insurance Specialist... Output *only* your response as an expert agent.",
)

# 2. El Agente Enrutador (Root)
root_agent = LlmAgent(
    name="InsuranceRouterAgent",
    model=GEMINI_MODEL,
    description="The main orchestrator that greets the user and routes the task.",
    sub_agents=[car_insurance_agent, life_insurance_agent, health_insurance_agent],
    instruction="""You are the Main Coordinator (Router) of an insurance agency.
    Your *only* job is to analyze the client's initial request and route it to the correct specialized sub-agent.

    Routing Rules:
    1. Analyze the user's intent.
    2. Review the 'descriptions' of your available sub-agents.
    3. Silently delegate the task to the best matching agent. DO NOT attempt to answer the insurance question yourself.""",
)
```

---

## 2. El Agente Secuencial: La Línea de Ensamblaje Determinista

Un agente de flujo de trabajo que ejecuta sus sub-agentes en un orden fijo y estricto. Es ideal para procesos donde cada paso depende del anterior.

### **¿Por qué es útil este patrón?**

* Ejecución en orden garantizado.
* Comportamiento determinístico.
* El estado se comparte entre sub-agentes.

El `SequentialAgent` ejecuta sus sub-agentes en un orden fijo y estricto. Es el patrón perfecto cuando cada paso del proceso depende del resultado (estado) del paso anterior.

### Casos de Uso Empresariales
* **Desarrollo de Software Automatizado:** Escribir código -> Revisar Código -> Refactorizar.
* **Marketing (Prospección B2B):** Extraer datos web -> Identificar *Pain Points* -> Redactar email persuasivo.

### El Código (`p02_sequential/agent.py`)

Aquí simulamos una pequeña fábrica de software autónoma:

```python
from google.adk.agents import LlmAgent, SequentialAgent

GEMINI_MODEL = "gemini-3.5-flash-lite"

code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model=GEMINI_MODEL,
    instruction="You are a Python Code Generator... Output *only* the complete Python code block.",
    output_key="generated_code",
)

code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model=GEMINI_MODEL,
    instruction="""You are an expert Python Code Reviewer.
    Review the code provided in {generated_code} for correctness, readability, and best practices.
    Output *only* the review comments or "No major issues found." """,
    output_key="review_comments",
)

code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Python Code Refactoring AI.
    Improve the {generated_code} based on the {review_comments}.
    Output *only* the refactored code.""",
    output_key="refactored_code",
)

# El orquestador secuencial
root_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
)
```

---

## 3. El Patrón de Bucle (`LoopAgent`): Auditoría y Perfección Iterativa

El `LoopAgent` es un orquestador que ejecuta una secuencia de sub-agentes de forma iterativa, ideal para tareas que requieren refinamiento o repetición.

### **Mecanismos de Terminación**

* `max_iterations`: Se establece un número máximo de ciclos para prevenir bucles infinitos.
* Condición de Salida: Un sub-agente evalúa el estado y, si se cumple un criterio, puede "escalar" una señal para detener el bucle.

Los modelos fundacionales no siempre aciertan a la primera. El `LoopAgent` introduce el concepto de **Auto-Reflexión** y mecanismos de terminación temprana. Un agente "Crítico" evalúa el trabajo de un "Creador" y lo envía a un "Refinador" hasta que se cumplan criterios estrictos de calidad.

### Casos de Uso Empresariales
* **Legal y Compliance:** Un contrato se redacta y audita en bucle hasta que cumple al 100% con la matriz de riesgo (GDPR, exclusividad, etc.).
* **Ingeniería de Datos:** Validación de extracción de JSON contra un esquema estricto, corrigiendo formatos de fecha y comas faltantes iterativamente.

### El Código (`p03_loop/agent.py`)

Implementamos un sistema de revisión de textos creativos usando una función Python como herramienta (`ToolContext`) para romper el bucle:

```python
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import ToolContext

GEMINI_MODEL = "gemini-3.5-flash-lite"
COMPLETION_PHRASE = "No major issues found."

# Herramienta para salir del bucle
def exit_loop(tool_context: ToolContext) -> dict:
    """Call this function ONLY when the critique indicates no further changes are needed."""
    tool_context.actions.escalate = True
    return {}

critic_agent_in_loop = LlmAgent(
    name="CriticAgent",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=f"""You are a Constructive Critic AI reviewing a short document draft: {{current_document}}
    IF you identify clear improvements, provide them.
    ELSE IF the document is coherent, respond *exactly* with the phrase "{COMPLETION_PHRASE}".""",
    output_key="criticism",
)

refiner_agent_in_loop = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_MODEL,
    include_contents="none",
    instruction=f"""You are a Creative Writing Assistant refining a document: {{current_document}}
    Analyze the {{criticism}}.
    IF the critique is exactly "{COMPLETION_PHRASE}", you MUST call the 'exit_loop' function.
    ELSE apply suggestions to improve the document.""",
    tools=[exit_loop],
    output_key="current_document",
)

# El Loop
refinement_loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[critic_agent_in_loop, refiner_agent_in_loop],
    max_iterations=5,  # Previene bucles infinitos
)
```

---

## 4. El Agente Paralelo: Velocidad y Escala Masiva

Ejecuta sub-agentes de forma concurrente para acelerar drásticamente los flujos de trabajo donde las tareas no dependen entre sí.

### **¿Qué ventajas tiene?**

* Velocidad y Eficiencia: Ideal para tareas intensivas e independientes como la recolección de datos de múltiples fuentes.
* Aislamiento de Estado: Cada sub-agente opera en su propia "rama" sin compartir estado automáticamente, previniendo conflictos.
* Orquestación Determinista: Aunque los sub-agentes son concurrentes, el `ParallelAgent` en sí mismo es un orquestador que no utiliza un LLM.

¿Por qué investigar secuencialmente si las tareas no dependen entre sí? El `ParallelAgent` (patrón *Scatter-Gather*) ejecuta múltiples agentes de forma concurrente en ramas aisladas y luego fusiona (sintetiza) los resultados. Es ideal para multiplicar la productividad.

### Casos de Uso Empresariales
* **Finanzas (Due Diligence):** Un agente extrae ganancias (Q3), otro busca escándalos en noticias, y otro analiza a la competencia, todo al mismo tiempo.
* **Inteligencia Competitiva:** Escanear simultáneamente los precios y *features* de 5 competidores web en segundos.

### El Código (`p04_parallel/agent.py`)

Lanzamos tres agentes investigadores concurrentes que utilizan una herramienta personalizada de búsqueda en Wikipedia (`search_wikipedia`), y un agente fusionador que consolida el reporte:

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

GEMINI_MODEL = "gemini-3.5-flash-lite"

def search_wikipedia(query: str) -> dict:
    """Search Wikipedia articles and return concise snippets for a query."""
    ...

research_tools = [search_wikipedia]

researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="Research 'renewable energy sources' using search_wikipedia. Summarize concisely.",
    tools=research_tools,
    output_key="renewable_energy_result",
)

researcher_agent_2 = LlmAgent(...)  # Similar para Vehículos Eléctricos
researcher_agent_3 = LlmAgent(...)  # Similar para Captura de Carbono

# Ejecución Concurrente
parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
)

# Agente de Síntesis
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""Synthesize the following research summaries into a structured report:
    - Renewable Energy: {renewable_energy_result}
    - Electric Vehicles: {ev_technology_result}
    - Carbon Capture: {carbon_capture_result}
    Ground your response *exclusively* on these inputs.""",
)

# Orquestador Final
root_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[parallel_research_agent, merger_agent],
)
```

---

## Conclusión

Migrar de simples *prompts* a **Patrones Agénticos** es lo que separa un experimento de IA de un sistema empresarial robusto y en producción. El ADK de Google elimina la fricción de la infraestructura, permitiéndote diseñar equipos virtuales que colaboran, iteran y actúan en paralelo.

La próxima vez que enfrentes un problema complejo, pregúntate: *¿Necesito un agente más inteligente, o necesito un mejor equipo de agentes?*
