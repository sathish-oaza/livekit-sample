import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, silero, azure, turn_detector


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Azure STT, OpenAI LLM and TTS plugins
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=azure.STT(
            speech_key="AZgPJ0wY7IqNucJyVa1CVXrqNwQ2ATEq0m5PKaqPOT3S4YSeZ2ySJQQJ99BAACHYHv6XJ3w3AAAAACOG8RRG",
            speech_region="eastus2",
            speech_host="https://eastus2.stt.speech.microsoft.com"
        ),
        llm=openai.LLM.with_azure(
            model="gpt-4o",
            azure_endpoint="https://ai-cmseusdev352332062725.cognitiveservices.azure.com",
            azure_deployment="gpt-4o",
            api_version="2024-08-01-preview",
            api_key="AZgPJ0wY7IqNucJyVa1CVXrqNwQ2ATEq0m5PKaqPOT3S4YSeZ2ySJQQJ99BAACHYHv6XJ3w3AAAAACOG8RRG"
        ),
        tts=azure.TTS(
            speech_key="AZgPJ0wY7IqNucJyVa1CVXrqNwQ2ATEq0m5PKaqPOT3S4YSeZ2ySJQQJ99BAACHYHv6XJ3w3AAAAACOG8RRG",
            speech_region="eastus2",
            speech_host="https://eastus2.tts.speech.microsoft.com"
        ),
        turn_detector=turn_detector.EOUModel(),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
