import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.filter import has_university_context
from src.hybrid import HybridRetriever
from src.loader import VERIFIED_STUDENT_PATH, read_records
from src.programs import classify_fee, identify_programs, tuition_programs
from src.service import ChatService
from src.student import classify_intents

SOURCE = "https://seu.edu.bd/uploads/media/2025/07/Admission_Prospectus_Fall_2025.pdf"
CASES = (
    ("Can you share the tuition fee of BBA?", ("BBA",), ("4,950", "127", "628,650", "658,650")),
    ("What is the tuition fee of CSE?", ("CSE",), ("4,750", "150", "712,500", "777,500")),
    ("BBA tuition fee koto?", ("BBA",), ("4,950", "127", "628,650", "658,650")),
    ("CSE er total cost koto?", ("CSE",), ("777,500",)),
    ("BBA এর টিউশন ফি কত?", ("BBA",), ("4,950", "127", "628,650", "658,650")),
    ("How much does EEE cost?", ("EEE",), ("3,450", "153", "527,850", "598,500")),
    ("What is the MBA per credit fee?", ("MBA_REGULAR", "MBA_ONE_YEAR"), ("4,200", "4,250")),
    ("What is the current BBA fee?", ("BBA",), ("could not verify current", "4,950", "658,650")),
    ("Compare CSE and BBA tuition fees.", ("CSE", "BBA"), ("4,750", "4,950", "777,500", "658,650")),
)


class TuitionTests(unittest.TestCase):
    def setUp(self):
        self.records = {record["id"]: record for record in read_records(VERIFIED_STUDENT_PATH)}
        docs = {key: SimpleNamespace(metadata=value, page_content=value["clean_text"])
                for key, value in self.records.items()}
        store = SimpleNamespace(index_to_docstore_id=dict(enumerate(docs)),
                                docstore=SimpleNamespace(search=docs.get))
        self.retriever = HybridRetriever(store)
        self.generate = Mock()
        self.chat = ChatService(self.retriever, self.generate)

    def test_program_queries_use_only_matching_official_rows(self):
        for question, programs, figures in CASES:
            with self.subTest(question=question):
                self.assertEqual(classify_intents(question)[0], "Tuition Fees and Payment")
                self.assertEqual(identify_programs(question), programs)
                self.assertEqual(tuition_programs(question), programs)
                matches = self.retriever.search_program_fees(programs)
                self.assertEqual({match.document.metadata["fee_schedule"]["program"] for match in matches}, set(programs))
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "unverified" if "current" in question else "university")
                self.assertTrue(all(value in reply.answer for value in (*figures, "Fall 2025")))
                self.assertEqual(len(reply.citations), len(programs))
                self.assertTrue(all(c["url"] == SOURCE for c in reply.citations))
                self.assertEqual([self.records[c["source_id"]]["fee_schedule"]["program"] for c in reply.citations],
                                 list(programs))
                if programs == ("BBA",):
                    self.assertNotIn("777,500", reply.answer)
        self.generate.assert_not_called()

    def test_no_verified_fee_means_no_figure(self):
        self.retriever.fee_documents = {}
        reply = self.chat.chat("What is the tuition fee of CSE?")
        self.assertEqual(reply.route, "unverified")
        self.assertNotIn("4,750", reply.answer)
        self.assertEqual(reply.citations[0]["source_id"], "seu-admission")
        self.generate.assert_not_called()

    def test_no_cross_session_fees(self):
        for question in ("CSE fees for Fall 2026?", "CSE tuition in Spring 2025?",
                         "স্প্রিং ২০২৫ এর সিএসই ফি কত?", "Compare CSE and B.Pharm tuition fees."):
            with self.subTest(question=question):
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "unverified")
                self.assertNotIn("4,750", reply.answer)
                self.assertNotIn("5,350", reply.answer)
        self.generate.assert_not_called()

    def test_other_university_cannot_inherit_seu_price(self):
        reply = self.chat.chat("What is the CSE tuition fee at BRAC University?")
        self.assertEqual(reply.route, "out_of_scope")
        self.assertFalse(reply.citations)
        self.assertNotIn("4,750", reply.answer)

    def test_student_topics_do_not_require_university_name(self):
        for question in ("Admission documents?", "Can I register for courses?", "How is CGPA calculated?",
                         "How do BBA waivers work?", "Where is the faculty directory?",
                         "When are exams?", "Where is the UMS portal?",
                         "MBA entry requirements?", "EEE academic advising?"):
            self.assertTrue(has_university_context(question), question)
        self.assertFalse(has_university_context("Who won the cricket match?"))
        self.assertFalse(has_university_context("What is the salary for a BBA graduate?"))
        self.assertFalse(has_university_context("What does architecture software cost?"))

    def test_student_question_with_weak_evidence_is_not_off_topic(self):
        self.generate.return_value = json.dumps({"status": "out_of_scope"})
        service = ChatService(Mock(search=Mock(return_value=[])), self.generate)
        self.assertEqual(service.chat("How is CGPA calculated?").route, "unverified")
        self.generate.assert_not_called()

    def test_separate_charges_and_program_variants(self):
        cases = (
            ("What is the BBA admission fee?", "15,000", "Fall 2025"),
            ("B.Pharm tuition fee?", "5,350", "Spring 2026"),
            ("MBA one-year cost?", "4,250", "Fall 2025"),
            ("EMBA fee?", "4,000", "Fall 2025"),
            ("MA in Bangla tuition?", "50,000", "Fall 2025"),
        )
        for question, figure, session in cases:
            with self.subTest(question=question):
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "university")
                self.assertIn(figure, reply.answer)
                self.assertEqual(reply.citations[0]["admission_session"], session)
        self.generate.assert_not_called()



    def test_precise_fee_requests(self):
        cases = (
            ("What is the admission fee of CSE?", "admission_fee", ("CSE",),
             ("15,000",), ("18,100", "11,400", "712,500", "777,500")),
            ("How much do I need to pay when joining CSE?", "initial_payment", ("CSE",),
             ("11,400", "18,100", "20%"), ("29,500", "712,500", "777,500")),
            ("What is the tuition per credit for CSE?", "tuition_per_credit", ("CSE",),
             ("4,750",), ("15,000", "712,500", "777,500")),
            ("What is the total cost of CSE?", "total_cost", ("CSE",),
             ("777,500",), ("15,000", "4,750", "712,500")),
            ("CSE admission fee koto?", "admission_fee", ("CSE",),
             ("15,000",), ("11,400", "18,100", "777,500")),
            ("CSE te vorti hote koto taka lagbe?", "initial_payment", ("CSE",),
             ("11,400", "18,100", "20%"), ("29,500", "712,500", "777,500")),
            ("What is the admission fee for BBA?", "admission_fee", ("BBA",),
             ("15,000",), ("18,100", "628,650", "658,650")),
            ("What is the application fee for CSE?", "application_fee", ("CSE",),
             ("1,000",), ("15,000", "18,100")),
            ("What is the total tuition for CSE?", "total_tuition", ("CSE",),
             ("712,500",), ("15,000", "777,500")),
            ("What is CSE semester tuition for 12 credits?", "semester_tuition", ("CSE",),
             ("57,000",), ("712,500", "777,500")),
            ("What is the initial tuition deposit for CSE?", "tuition_deposit", ("CSE",),
             ("11,400",), ("15,000", "18,100", "777,500")),
        )
        for question, kind, programs, included, excluded in cases:
            with self.subTest(question=question):
                self.assertEqual(classify_fee(question), (kind, programs))
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "university")
                self.assertIn("Fall 2025", reply.answer)
                self.assertTrue(all(text in reply.answer for text in included), reply.answer)
                self.assertTrue(all(text not in reply.answer for text in excluded), reply.answer)
                self.assertTrue(reply.citations)
                for citation in reply.citations:
                    self.assertEqual(citation["url"], SOURCE)
                    self.assertIn(kind, self.records[citation["source_id"]]["fee_type"])
        self.generate.assert_not_called()

    def test_missing_precise_fee_does_not_use_program_total(self):
        for question in ("What is the CSE admission fee for Fall 2026?", "What is CSE semester tuition?"):
            with self.subTest(question=question):
                reply = self.chat.chat(question)
                self.assertEqual(reply.route, "unverified")
                self.assertNotIn("777,500", reply.answer)
                self.assertNotIn("15,000", reply.answer)
        self.generate.assert_not_called()
