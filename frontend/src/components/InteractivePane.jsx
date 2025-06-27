import React from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import ChatPanel from './ChatPanel';
import CodePanel from './CodePanel';
import Quiz from './Quiz';
import { MessageCircle, Code, HelpCircle } from 'lucide-react';

const InteractivePane = ({ lessonId }) => {
  return (
    <div className="w-96 bg-card border-l border-border h-full flex flex-col">
      <div className="p-4 border-b border-border">
        <h2 className="text-lg font-semibold">Interactive Learning</h2>
      </div>
      
      <Tabs defaultValue="chat" className="flex-1 flex flex-col">
        <TabsList className="grid w-full grid-cols-3 m-4 mb-2">
          <TabsTrigger value="chat" className="flex items-center gap-2">
            <MessageCircle className="h-4 w-4" />
            Chat
          </TabsTrigger>
          <TabsTrigger value="code" className="flex items-center gap-2">
            <Code className="h-4 w-4" />
            Code Lab
          </TabsTrigger>
          <TabsTrigger value="quiz" className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            Quiz
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="chat" className="flex-1 m-4 mt-0">
          <ChatPanel lessonId={lessonId} />
        </TabsContent>
        
        <TabsContent value="code" className="flex-1 m-4 mt-0">
          <CodePanel lessonId={lessonId} />
        </TabsContent>
        
        <TabsContent value="quiz" className="flex-1 m-4 mt-0">
          <Quiz lessonId={lessonId} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default InteractivePane; 