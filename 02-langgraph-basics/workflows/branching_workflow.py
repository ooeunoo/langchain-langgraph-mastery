from typing import Literal, TypedDict, cast
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from common.llm import get_llm
from utils.logger import setup_logger

logger = setup_logger("branching_workflow")

class TicketAnalysis(BaseModel):
    ticket_type: Literal['technical', 'billing', 'general'] = Field(
        description="Ticket type: technical, billing, or general"
    )
    priority: Literal['low', 'medium', 'high', 'urgent'] = Field(
        description='Priority level'
    )
    summary: str = Field(description="Summary of ticket content")
    keywords: list[str] = Field(description="Extracted keywords")

class TechnicalSolution(BaseModel):
    """Technical support solution"""
    diagnosis: str = Field(description="Problem diagnosis")
    solution_steps: list[str] = Field(description="Solution steps")
    estimated_time: str = Field(description="Estimated time required")
    requires_escalation: bool = Field(description="Requires escalation to senior")

class BillingSolution(BaseModel):
    """Billing solution"""
    issue_type: str = Field(description="Billing issue type")
    amount_involved: str = Field(description="Amount involved")
    resolution: str = Field(description="Resolution plan")
    refund_eligible: bool = Field(description="Refund eligible")

class GeneralResponse(BaseModel):
    """General inquiry response"""
    category: str = Field(description="Inquiry category")
    response: str = Field(description="Response content")
    helpful_links: list[str] = Field(description="Helpful links")
    follow_up_needed: bool = Field(description="Follow-up needed")

class TicketState(TypedDict):
    """Ticket processing state"""
    ticket_id: str
    customer_message: str
    analysis: TicketAnalysis | None
    technical_solution: TechnicalSolution | None
    billing_solution: BillingSolution | None
    general_response: GeneralResponse | None
    final_response: str
    status: str

def analyze_ticket(state: TicketState) -> TicketState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(TicketAnalysis)

    prompt = f"""
Analyze the following customer ticket:

Ticket ID: {state['ticket_id']}
Customer Message: {state['customer_message']}

Classify the ticket type as one of:
- technical: Technical issues (login errors, features not working, etc.)
- billing: Billing/payment related (charges, refund requests, etc.)
- general: General inquiries (usage questions, information requests, etc.)

Priority levels: low, medium, high, urgent
"""

    analysis = cast(TicketAnalysis, structured_llm.invoke(prompt))

    return TicketState(analysis=analysis, status="Analysis complete") # type: ignore

def handle_technical(state: TicketState) -> TicketState:
    analysis = state["analysis"]
    if not analysis:
        return TicketState(status="No analysis") # type: ignore
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(TechnicalSolution)

    prompt = f"""
Diagnose and provide a solution for this technical problem:

Customer Message: {state['customer_message']}
Analysis Summary: {analysis.summary}
Keywords: {', '.join(analysis.keywords)}

Provide specific diagnosis and step-by-step solution.
"""

    solution = cast(TechnicalSolution, structured_llm.invoke(prompt))
    
    return TicketState(technical_solution=solution, status="Technical support complete") # type: ignore

def handle_billing(state: TicketState) -> TicketState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(BillingSolution)

    prompt = f"""
Handle this billing inquiry:

Customer Message: {state['customer_message']}

Identify the billing issue and provide a resolution.
Determine if a refund is eligible.
"""

    solution = cast(BillingSolution, structured_llm.invoke(prompt))

    return TicketState(billing_solution=solution, status="Billing processing complete") # type: ignore

def handle_general(state: TicketState) -> TicketState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(GeneralResponse)

    prompt = f"""
Respond to this general inquiry:

Customer Message: {state['customer_message']}

Provide a friendly and clear answer.
Recommend helpful links or resources.
"""

    response = cast(GeneralResponse, structured_llm.invoke(prompt))

    return TicketState(general_response=response, status="General inquiry complete") # type: ignore

def generate_final_response(state: TicketState) -> TicketState:
    if state.get('technical_solution'):
        solution = state["technical_solution"]
        if not solution:
            return TicketState(final_response="Error processing technical solution", status="Error") # type: ignore

        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(solution.solution_steps)])
        escalation_note = "This issue requires additional support - a specialist will be assigned." if solution.requires_escalation else "Please try these steps and contact us if the issue persists."

        response = f"""Hello, technical support for ticket {state['ticket_id']}.

Problem Diagnosis:
{solution.diagnosis}

Solution Steps:
{steps_text}

Estimated Time: {solution.estimated_time}

{escalation_note}

Thank you.
"""

    elif state.get('billing_solution'):
        solution = state['billing_solution']
        if not solution:
            return TicketState(final_response="Error processing billing solution", status="Error") # type: ignore

        refund_note = "Refund process will begin. It takes 3-5 business days." if solution.refund_eligible else "Unfortunately, a refund is not available in this case."

        response = f"""Hello, billing response for ticket {state['ticket_id']}.

Issue Type: {solution.issue_type}
Amount Involved: {solution.amount_involved}

Resolution:
{solution.resolution}

{refund_note}

Please contact us if you have additional questions.
"""

    elif state.get('general_response'):
        resp = state['general_response']
        if not resp:
            return TicketState(final_response="Error processing general response", status="Error") # type: ignore

        links_text = "\n".join([f"- {link}" for link in resp.helpful_links])
        follow_up = "A representative will contact you for additional assistance." if resp.follow_up_needed else ""

        response = f"""Hello, response to ticket {state['ticket_id']}.

Category: {resp.category}

Response:
{resp.response}

Helpful Resources:
{links_text}

{follow_up}

Thank you!
"""
    else:
        response = "An error occurred during processing."


    return TicketState(final_response=response, status="Complete") # type: ignore

def route_by_ticket_type(state: TicketState) -> Literal['technical', 'billing', 'general']:
    analysis = state.get("analysis")

    if not analysis:
        return "general"
    
    ticket_type = analysis.ticket_type

    return ticket_type


def create_branching_workflow():
    builder = StateGraph(TicketState)

    builder.add_node("analyze", analyze_ticket)
    builder.add_node("technical", handle_technical)
    builder.add_node("billing", handle_billing)
    builder.add_node("general", handle_general)
    builder.add_node("finalize", generate_final_response)

    builder.add_edge(START, "analyze")

    builder.add_conditional_edges(
        "analyze",
        route_by_ticket_type,
        {
            "technical": "technical",
            "billing": "billing",
            "general": "general"
        }
    )

    # Each path -> finalize
    builder.add_edge("technical", "finalize")
    builder.add_edge("billing", "finalize")
    builder.add_edge("general", "finalize")

    # Finalize -> end
    builder.add_edge("finalize", END)
    
    memory = MemorySaver()

    return builder.compile(checkpointer=memory)

def branching_workflow_example():
    graph = create_branching_workflow()

    ########################### 1
    state1 = TicketState(
        ticket_id="TECH-1",
        customer_message="I keep getting 'Invalid credentials' error when trying to login. I've reset my password but the same issue persists.",
        analysis=None,
        technical_solution=None,
        billing_solution=None,
        general_response=None,
        final_response="",
        status="Waiting"
    ) # type: ignore

    config1 = RunnableConfig(configurable={"thread_id": "tech-1"})

    result1 = graph.invoke(state1, config1)
    logger.info("\nFinal Response:")
    logger.info(result1['final_response'])
    logger.info("="*60)

    ############################## 2

    state2 = TicketState(
        ticket_id="BILL-2",
        customer_message="I see a duplicate charge on last month's bill. $49.99 was charged twice. Can I get a refund?",
        analysis=None,
        technical_solution=None,
        billing_solution=None,
        general_response=None,
        final_response="",
        status="Waiting"
    )

    config2 = RunnableConfig(configurable={"thread_id": "bill-2"})

    result2 = graph.invoke(state2, config2)
    logger.info("\nFinal Response:")
    logger.info(result2['final_response'])
    logger.info("="*60)

    ############################## 2

    state3 = TicketState(
        ticket_id="GEN-3",
        customer_message="I'd like to learn more about the premium plan features. What are the differences from the free plan?",
        analysis=None,
        technical_solution=None,
        billing_solution=None,
        general_response=None,
        final_response="",
        status="Waiting"
    )

    config3 = RunnableConfig(configurable={"thread_id": "gen-0"})

    result3 = graph.invoke(state3, config3)
    logger.info("\nFinal Response:")
    logger.info(result3['final_response'])

if __name__ == "__main__":
    branching_workflow_example()
