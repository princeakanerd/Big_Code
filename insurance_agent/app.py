import streamlit as st
import fitz  # PyMuPDF
import re
import os
from main import app as agent_app

st.set_page_config(layout="wide", page_title="AI Claims Adjudicator")

def render_pdf_with_highlights(pdf_path, chunks):
    """Opens a PDF, searches for the exact text contained in the chunks, and highlights them."""
    if not os.path.exists(pdf_path):
        st.error(f"Could not find PDF at {pdf_path}")
        return None
        
    try:
        doc = fitz.open(pdf_path)
        pages_to_render = set()
        
        # Iterate over all chunks found by the state graph
        for chunk in chunks:
            # Parse out the page number and text
            # Format: "[Page 1]: Cardiovascular Services Section..."
            match = re.match(r"\[Page (\w+)\]:\s*(.*)", chunk)
            if match:
                page_str, text_to_find = match.groups()
                
                # Assume page string represents a 1-indexed number
                page_num = 0
                if page_str.isdigit():
                    page_num = int(page_str) - 1 # PyMuPDF is 0-indexed
                
                # Protect bounds
                if 0 <= page_num < len(doc):
                    page = doc[page_num]
                    pages_to_render.add(page_num)
                    
                    # Highlight the targeted text chunk exactly
                    # Strip to help search matching
                    search_term = text_to_find[:50].strip() # Check first portion of the matched sub-clause
                    text_instances = page.search_for(search_term)
                    
                    if not text_instances and len(search_term.split()) > 0:
                        # Fallback to the first word
                        text_instances = page.search_for(search_term.split()[0])
                        
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        if highlight:
                            highlight.set_colors(stroke=(1, 1, 0)) # Yellow box
                            highlight.update()
        
        rendered_images = []
        for page_num in sorted(pages_to_render):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Zoom for high-res
            img_data = pix.tobytes("png")
            rendered_images.append((page_num + 1, img_data))
            
        return rendered_images if rendered_images else None
        
    except Exception as e:
        st.error(f"Error rendering PDF visually: {e}")
        return None


# --- DASHBOARD LAYOUT ---
st.title("Human-in-the-Loop Insurance Adjudication")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Claim Intake & Decision")
    
    with st.form("claim_form"):
        raw_text = st.text_area(
            "Incoming Hospital Bill Extract", 
            "Patient underwent a cardiovascular stress test (CPT-93015) on Oct 14th. Total cost: $450.00. Diagnosis: suspected arrhythmia."
        )
        submit = st.form_submit_button("Run AI Adjudication")

    if submit:
        with st.spinner("Executing LangGraph Multi-Agent Workflows..."):
            initial_state = {"claim_details": {"raw_text": raw_text}}
            
            try:
                final_state = agent_app.invoke(initial_state)
                # Store in session state so it persists across button clicks
                st.session_state['final_state'] = final_state
                st.success("Workflow Complete!")
            except Exception as e:
                st.error(f"Error executing agent workflow: {e}")

    # Display results if available
    if 'final_state' in st.session_state:
        state = st.session_state['final_state']
        
        st.subheader("AI Draft Decision")
        st.info(state.get("draft_decision", "No decision rendered."))
        
        verifier_errors = state.get("errors", [])
        if verifier_errors:
            st.warning(f"Verifier caught hallucinations during run: {verifier_errors}")
            
        st.write("---")
        st.subheader("Human Validation Controls")
        col_app, col_rej, col_esc = st.columns(3)
        
        with col_app:
            if st.button("✅ Approve AI Action", use_container_width=True):
                st.success("Claim validated and finalized.")
        with col_rej:
            if st.button("❌ Reject AI Action", use_container_width=True):
                st.error("Claim denied. AI instructions negated.")
        with col_esc:
            if st.button("⚠️ Escalate Review", use_container_width=True):
                st.warning("Flagged for Senior Adjuster Review.")

with col2:
    st.header("Visual Grounding")
    
    if 'final_state' in st.session_state:
        chunks = st.session_state['final_state'].get("retrieved_policy_chunks", [])
        if chunks:
            st.markdown("**(System automatically overlaid reasoning based on exact text attribution)**")
            pdf_path = "data/sample_policy.pdf" # Path relative to where we run streamit
            images = render_pdf_with_highlights(pdf_path, chunks)
            
            if images:
                for page_no, img_data in images:
                    st.image(img_data, caption=f"Source Document - Page {page_no} Highlighting Citations", use_container_width=True)
            else:
                st.info("Could not visually locate precise coordinates for highlight in PDF layer.")
        else:
            st.info("No policy clauses were retrieved for visual grounding.")
    else:
        st.info("Submit a claim down the left column to generate visually grounded citations.")
