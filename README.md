# Fundamentals of Prompting and Chaining

**Session Time:** 120 minutes

---

## Table of Contents

1. Beyond Raw APIs: The Case for Orchestration
2. Deep Dive: Advanced Prompt Templates
3. The Glue of AI Engineering: Chains and LCEL
4. Lab: Building Sequential Chains (The "Chain Reactor")
5. Wrap-Up and Reflection
6. Resources

---

## Learning Objectives

By the end of this module, participants will be able to:

1.  **Abstract** direct LLM API calls using LangChain's model wrappers (`ChatOpenAI`, `HuggingFaceEndpoint`).
2.  **Engineer** reusable, dynamic `PromptTemplates` that incorporate advanced techniques like Few-Shot prompting directly into the data structure.
3.  **Construct** modular AI pipelines using the LangChain Expression Language (LCEL) to pipe outputs from one component to another.
4.  **Implement** a sequential chain that performs task decomposition (e.g., generating a list, then processing each item).

---

## Session Breakdown

| Segment | Topic | Duration (minutes) |
| :--- | :--- | :--- |
| 1 | Beyond Raw APIs: Introduction to LangChain | 15 |
| 2 | Deep Dive: Advanced Prompt Templates | 25 |
| 3 | The Glue: Chains and LCEL (LangChain Expression Language) | 30 |
| 4 | Lab: Building Sequential Chains | 40 |
| 5 | Wrap-Up and Reflection | 10 |

---

## 1. Beyond Raw APIs: The Case for Orchestration (15 min)

* **The Problem with Raw `requests`:**
    * While calling `openai.ChatCompletion.create` is easy for one-off tasks, building an *application* requires more: managing conversation history, switching models without rewriting code, parsing outputs uniformly, and handling retries.
* **LangChain as Infrastructure:**
    * LangChain is not just a wrapper; it's an orchestration framework. It standardizes the interface for different LLMs (OpenAI, Anthropic, local Llama via Ollama).
    * *Key Abstraction:* The Chat Model object.
        ```python
        from langchain_openai import ChatOpenAI
        
        # standardized interface regardless of the underlying provider
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        response = llm.invoke("Explain standard deviation to a 5th grader.")
        print(response.content) 
        ```

## 2. Deep Dive: Advanced Prompt Templates (25 min)

Hardcoding prompt strings using Python f-strings (e.g., `f"Translate {text} to French"`) doesn't scale when prompts become 50 lines long with multiple complex variables.

* **The `PromptTemplate` Object:**
    * Separates the *structure* of the prompt from the *data* it needs.
    * Validates missing input variables before sending to the LLM, preventing expensive errors.
* **ChatPromptTemplate (Handling Roles):**
    * Crucial for modern chat models, organizing inputs into System, Human, and AI messages.
    * *Technique: Baked-in Few-Shot Prompting:*
        ```python
        from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

        # 1. Define examples
        examples = [
            {"input": "happy", "output": "ecstatic"},
            {"input": "sad", "output": "despondent"}
        ]
        
        # 2. Define how each example should be formatted
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        
        # 3. Create the few-shot template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        
        # 4. Final combined prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a thesaurus that only gives extreme synonyms."),
            few_shot_prompt,
            ("human", "{input}")
        ])
        ```

## 3. The Glue: Chains and LCEL (30 min)

* **What is a Chain?**
    * A sequence of calls to components, which can include other chains. Most common: `Prompt -> Model -> OutputParser`.
* **LCEL (LangChain Expression Language):**
    * The modern, declarative way to compose chains using the unix pipe `|` operator.
    * *Why it's superior to legacy `LLMChain`:* It automatically handles streaming, async calls, and batching out of the box.
* **Core Components of LCEL:**
    * `Runnable`: Almost everything in LangChain is a "Runnable" that implements standard methods like `.invoke()`, `.stream()`, and `.batch()`.
    * *The Standard Pattern:*
        ```python
        from langchain_core.output_parsers import StrOutputParser
        
        # The components
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        model = ChatOpenAI(model="gpt-3.5-turbo")
        output_parser = StrOutputParser() # Extracts just the .content string from the AI message
        
        # The Chain
        chain = prompt | model | output_parser
        
        # Execution
        result = chain.invoke({"topic": "bears"})
        print(result)
        ```

## 4. Lab: Building Sequential Chains (40 min)

**Objective:** Build a "Chain Reactor" that takes a raw, messy brain-dump of text, extracts action items, and then drafts an email assigning those tasks. This requires passing the output of Chain A as the input to Chain B.

**Refer:** [Build Your First Chain](https://git.generalassemb.ly/applied-ai-and-deep-learning-in-action/Lab-4_1-Build-your-First-Chain)

## 5. Wrap-Up and Reflection (10 min)

**Reflection Questions:**
1.  Why did we use `StrOutputParser()` in our chains? What would happen if we removed it and piped the raw `AIMessage` object into the next prompt?
2.  How does LCEL's `|` syntax make complex pipelines easier to read compared to deeply nested function calls?
3.  In a sequential chain, if the first model hallucinates and outputs garbage, the second model will likely also output garbage. How might we add a "verification" step in the middle?

---

## 6. Resources

* [LangChain LCEL Documentation](https://python.langchain.com/docs/expression_language/) - *The official guide to modern chaining.*
* [LangSmith Tracing](https://smith.langchain.com/) - *Essential for debugging complex chains (visualizing the sequence).*
* [Prompt Engineering Guide](https://www.promptingguide.ai/) - *A comprehensive resource for advanced prompting strategies that can be baked into templates.*