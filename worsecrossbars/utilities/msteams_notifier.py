"""msteams_notifier:
An internal module used to send notifications
to the user via Microsoft Teams.
"""
import pymsteams


class MSTeamsNotifier:
    """A handler used to store the Webhook URL and provide messagging features.

    Args:
      webhook: The Webhook URL.
    """

    def __init__(self, webhook):
        self.webhook_url = webhook

    def __call__(self):
        pass

    def send_message(self, message: str, title: str = None, color: str = None):
        """Send a message to the channel indicated by the Webhook URL.

        Args:
          message: The message to send.
          title: An optional title.
          color: The color of the card.
        """

        msteams_message = pymsteams.connectorcard(self.webhook_url)
        msteams_message.text(message)
        if title:
            msteams_message.title(title)
        if color:
            msteams_message.color(color)
        msteams_message.send()
