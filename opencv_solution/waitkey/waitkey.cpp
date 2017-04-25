// waitkey.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;

char mywaitkay(int second = 0) {
	cout << "waiting key for " << second << " seconds" << endl;
	HANDLE stdinHandle = GetStdHandle(STD_INPUT_HANDLE);
	time_t startTime = time(NULL);
	while (time(NULL) < startTime + second) {
		if (WaitForSingleObject(stdinHandle, 100) == WAIT_OBJECT_0)
		{
			INPUT_RECORD record;
			DWORD numRead;
			if (!ReadConsoleInput(GetStdHandle(STD_INPUT_HANDLE), &record, 1, &numRead)) {
				// hmm handle this error somehow...
				continue;
			}

			if (record.EventType != KEY_EVENT) {
				// don't care about other console events
				continue;
			}

			if (!record.Event.KeyEvent.bKeyDown) {
				// really only care about keydown
				continue;
			}
			char res = record.Event.KeyEvent.uChar.AsciiChar;
			if (res < 'a' || 'z' < res) {
				continue;
			}

			// if you're setup for ASCII, process this:
			//record.Event.KeyEvent.uChar.AsciiChar
			return res;
		}
	}
	return 0;
}


int main()
{
	char c = mywaitkay(10);
	cout << c << endl;
    return 0;
}

