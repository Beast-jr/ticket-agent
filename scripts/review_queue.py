"""Human review queue: see escalated tickets that are waiting, and approve / edit / reject them.

Run from the project root:
  python scripts/review_queue.py list
  python scripts/review_queue.py show T12345
  python scripts/review_queue.py approve T12345 --reply "Thanks, please send the logs..."
  python scripts/review_queue.py approve T12345            (no --reply: accepts the rejected draft, if there is one)
  python scripts/review_queue.py edit T12345               (opens the rejected draft in Notepad to edit)
  python scripts/review_queue.py edit T12345 --reply "Corrected text"
  python scripts/review_queue.py reject T12345 --note "spam"

--by NAME says who decided (default: your Windows user name). It is stored with the time.
Approved and edited replies are added to the knowledge base (source 'learned'), if the ticket's run
was started with learning switched on. No LLM is called here.
"""
import argparse
import getpass
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.checkpoint import get_checkpointer
from agent.graph import get_saved_state, list_paused, resume_ticket

sys.stdout.reconfigure(encoding="utf-8")


def edit_in_notepad(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    editor = os.environ.get("EDITOR", "notepad")
    subprocess.run([editor, path], check=True)  # waits until the editor window is closed
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    finally:
        os.unlink(path)


def cmd_list(saver, args):
    paused = list_paused(saver)
    if not paused:
        print("No tickets are waiting for review.")
        return
    print(f"{len(paused)} ticket(s) waiting for review:\n")
    for p in paused:
        text = " ".join(p["text"].split())[:70]
        print(f"  {p['ticket_id']:<12} {p['category'] or '-':<26} {p['priority'] or '-':<7} {text!r}")
    print("\nSee one in full:  python scripts/review_queue.py show <id>")


def waiting_ticket(saver, ticket_id):
    snap = get_saved_state(ticket_id, saver)
    if snap.next != ("human_review",):
        sys.exit(f"Ticket {ticket_id!r} is not waiting for review (not found, or already decided).")
    return snap.values


def cmd_show(saver, args):
    v = waiting_ticket(saver, args.ticket_id)
    print(f"TICKET {args.ticket_id}   category: {v.get('category')}   priority: {v.get('priority')}\n")
    print("TICKET TEXT:\n" + v["text"] + "\n")
    print(v["escalation_summary"])


def cmd_decide(saver, args):
    v = waiting_ticket(saver, args.ticket_id)
    reply = (getattr(args, "reply", None) or "").strip()  # reject has no --reply option
    if args.command == "approve" and not reply:
        reply = v.get("rejected_reply", "")
        if not reply:
            sys.exit("This ticket has no draft to accept. Write a reply with --reply \"...\".")
        print("Accepting the rejected draft as it is.")
    elif args.command == "edit" and not reply:
        draft = v.get("rejected_reply", "")
        reply = edit_in_notepad(draft)
        if not reply:
            sys.exit("The edited reply is empty, nothing was changed.")
    decision = {"action": args.command, "reply": reply, "decided_by": args.by, "note": args.note or ""}
    try:
        final = resume_ticket(args.ticket_id, decision, saver)
    except ValueError as e:
        sys.exit(f"Not accepted: {e}")
    print(f"Done. Status is now {final['status']!r}.")
    if final["status"] == "replied_by_human":
        print("Reply:", final["final_reply"])
    if final.get("learned"):
        print(f"Added to the knowledge base as {final['learned_ticket_id']} (source 'learned').")


def main():
    parser = argparse.ArgumentParser(description="Review escalated tickets.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list tickets waiting for review")
    show = sub.add_parser("show", help="show one waiting ticket in full")
    show.add_argument("ticket_id")
    for name, text in [("approve", "send a reply written by you (or accept the rejected draft)"),
                       ("edit", "edit the rejected draft and send it"),
                       ("reject", "close the ticket with no reply")]:
        p = sub.add_parser(name, help=text)
        p.add_argument("ticket_id")
        if name != "reject":
            p.add_argument("--reply", help="the reply text")
        p.add_argument("--by", default=getpass.getuser(), help="who is deciding (default: your user name)")
        p.add_argument("--note", help="optional note stored with the decision")
    args = parser.parse_args()

    saver = get_checkpointer()
    try:
        {"list": cmd_list, "show": cmd_show}.get(args.command, cmd_decide)(saver, args)
    finally:
        saver.conn.close()


if __name__ == "__main__":
    main()
