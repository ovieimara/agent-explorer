import asyncio


class FormOrchestrator:
    def __init__(self, rag_agent, label_agent, fill_agent):
        self.rag_agent = rag_agent
        self.label_agent = label_agent
        self.fill_agent = fill_agent
        self.chain = self._create_async_chain()

    def _create_async_chain(self):
        from langchain_core.runnables import RunnableGenerator

        async def async_chain(state: dict) -> dict:
            # Get page from state
            page = state["page"]

            # 1. Locate fields (async)
            state["fields"] = await self.label_agent.locate_fields(page)

            # 2. Prepare questions
            state["questions"] = [
                field["label"] for field in state["fields"]
            ]

            # 3. Get RAG responses (async)
            state["responses"] = await asyncio.gather(*[
                self.rag_agent.get_field_value(question)
                for question in state["questions"]
            ])

            # 4. Process responses
            state["answers"] = [
                {"selector": field["xpath"], "value": response["value"]}
                for field, response in zip(state["fields"], state["responses"])
            ]

            # 5. Fill form (async)
            await self.fill_agent.fill_form(page, state["answers"])

            # 6. Handle submission (async)
            state["should_continue"] = await self.fill_agent.handle_submission(page)

            return state

        return RunnableGenerator(async_chain)

    async def process_form(self, page: Page) -> dict:
        """Execute the async workflow"""
        return await self.chain.ainvoke({"page": page})
